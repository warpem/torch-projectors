/**
 * CPU Kernels for Differentiable 2D Projection Operations in Fourier Space
 * 
 * This file implements high-performance CPU kernels for forward and backward projection
 * operations used in cryo-electron microscopy and tomography. All operations work in
 * Fourier space following the Projection-Slice Theorem.
 * 
 * Key Concepts:
 * - Forward projection: Sample from 3D Fourier reconstruction to create 2D projections
 * - Backward projection: Scatter 2D projection data into 3D Fourier reconstruction
 * - FFTW format: Real-to-complex FFT with last dimension N/2+1
 * - Friedel symmetry: F(k) = conj(F(-k)) for real-valued reconstructions
 * - Interpolation: Bilinear and bicubic methods with analytical gradients
 */

#include "projection_2d_kernels.h"
#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <complex>
#include <algorithm>
#include <atomic>
#include <omp.h>

/**
 * Atomic accumulation for complex numbers using separate real/imaginary parts
 * 
 * Since std::atomic<std::complex<T>> is not available, we treat complex numbers
 * as pairs of atomic floats and accumulate real/imaginary parts separately.
 * This avoids race conditions when multiple threads write to the same location.
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline void atomic_add_complex(scalar_t* target, const scalar_t& value) {
    // Cast to atomic real types for thread-safe accumulation
    std::atomic<real_t>* real_ptr = reinterpret_cast<std::atomic<real_t>*>(target);
    std::atomic<real_t>* imag_ptr = real_ptr + 1;
    
    // Atomically add real and imaginary parts
    real_ptr->fetch_add(value.real(), std::memory_order_relaxed);
    imag_ptr->fetch_add(value.imag(), std::memory_order_relaxed);
}

/**
 * Atomic accumulation for real numbers
 */
template <typename real_t>
inline void atomic_add_real(real_t* target, const real_t& value) {
    std::atomic<real_t>* atomic_ptr = reinterpret_cast<std::atomic<real_t>*>(target);
    atomic_ptr->fetch_add(value, std::memory_order_relaxed);
}

/**
 * Sample from FFTW-formatted Fourier space with automatic Friedel symmetry handling
 * 
 * FFTW real-to-complex format stores only positive frequencies in the last dimension.
 * For negative frequencies, we use Friedel symmetry: F(-kx,-ky) = conj(F(kx,ky))
 * 
 * @param rec: 3D complex tensor [batch, height, width/2+1] in FFTW format
 * @param b: batch index
 * @param boxsize: full size of the reconstruction (width before RFFT)
 * @param boxsize_half: width/2+1 (actual stored width)
 * @param r: row coordinate (can be negative, handled via wrapping)
 * @param c: column coordinate (can be negative, handled via Friedel symmetry)
 * @return: Complex value at (r,c) with proper symmetry handling
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline scalar_t sample_fftw_with_conjugate(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
    const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
    int64_t r, int64_t c
) {
    bool need_conjugate = false;

    // Handle negative kx via Friedel symmetry (c < 0)
    // For real-valued reconstructions: F(-kx,-ky) = conj(F(kx,ky))
    if (c < 0) {
        c = -c;          // Mirror to positive kx
        r = -r;          // ky must be mirrored as well for correct symmetry
        need_conjugate = !need_conjugate;
    }

    // Clamp coordinates to valid array bounds
    c = std::min(c, boxsize_half - 1);  // Column: [0, boxsize/2]
    r = std::min(boxsize / 2, std::max(r, -boxsize / 2 + 1));  // Row: [-boxsize/2+1, boxsize/2]

    // Convert negative row indices to positive (FFTW wrapping)
    // Negative frequencies are stored at the end of the array
    if (r < 0)
        r = boxsize + r;

    r = std::min(r, boxsize - 1);  // Final bounds check

    // Return conjugated value if we used Friedel symmetry
    if (need_conjugate)
        return std::conj(rec[b][r][c]);
    else
        return rec[b][r][c];
}



/**
 * Interpolation method enumeration
 * Currently supports bilinear and bicubic interpolation
 */
enum class InterpolationMethod {
    LINEAR,  // Bilinear interpolation (2x2 grid)
    CUBIC    // Bicubic interpolation (4x4 grid)
};

/**
 * Abstract base class for interpolation kernels
 * 
 * Provides a unified interface for different interpolation methods.
 * Each kernel must implement both forward interpolation and gradient computation.
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct InterpolationKernel {
    virtual ~InterpolationKernel() = default;
    
    /**
     * Perform interpolation at continuous coordinates (r,c)
     * @param rec: Source tensor to sample from
     * @param b: Batch index
     * @param boxsize: Full reconstruction size
     * @param boxsize_half: Half size (for FFTW format)
     * @param r: Row coordinate (floating point)
     * @param c: Column coordinate (floating point)
     * @return: Interpolated complex value
     */
    virtual scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const = 0;
    
    /**
     * Perform interpolation with simultaneous gradient computation
     * Used for computing gradients w.r.t. rotation parameters
     * @return: Tuple of (value, grad_r, grad_c)
     */
    virtual std::tuple<scalar_t, scalar_t, scalar_t> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const = 0;
};

/**
 * Bilinear interpolation kernel
 * 
 * Samples a 2x2 grid of neighboring pixels and performs bilinear interpolation.
 * This is the standard method used in most computer vision applications.
 * 
 * Mathematical formula:
 * f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct BilinearKernel : public InterpolationKernel<scalar_t, real_t> {
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const override {
        // Extract integer and fractional parts of coordinates
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);
        const real_t c_frac = c - c_floor;  // Fractional part [0,1)
        const real_t r_frac = r - r_floor;  // Fractional part [0,1)

        // Sample 2x2 grid of neighboring pixels
        // p00 = bottom-left, p01 = bottom-right, p10 = top-left, p11 = top-right
        const scalar_t p00 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor, c_floor);
        const scalar_t p01 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor, c_floor + 1);
        const scalar_t p10 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor);
        const scalar_t p11 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor + 1);

        // Bilinear interpolation: first interpolate horizontally, then vertically
        const scalar_t p0 = p00 + (p01 - p00) * c_frac;  // Bottom edge
        const scalar_t p1 = p10 + (p11 - p10) * c_frac;  // Top edge
        return p0 + (p1 - p0) * r_frac;  // Final vertical interpolation
    }
    
    std::tuple<scalar_t, scalar_t, scalar_t> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const override {
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);

        const real_t c_frac = c - c_floor;
        const real_t r_frac = r - r_floor;

        const scalar_t p00 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor, c_floor);
        const scalar_t p01 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor, c_floor + 1);
        const scalar_t p10 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor);
        const scalar_t p11 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor + 1);

        // Value computation (same as interpolate method)
        const scalar_t p0 = p00 + (p01 - p00) * c_frac;
        const scalar_t p1 = p10 + (p11 - p10) * c_frac;
        const scalar_t val = p0 + (p1 - p0) * r_frac;

        // Analytical spatial gradients derived from bilinear formula
        // ∂f/∂r and ∂f/∂c computed analytically for efficiency
        const scalar_t grad_r = (1 - c_frac) * (p10 - p00) + c_frac * (p11 - p01);
        const scalar_t grad_c = (1 - r_frac) * (p01 - p00) + r_frac * (p11 - p10);

        return std::make_tuple(val, grad_r, grad_c);
    }
};

/**
 * Bicubic interpolation kernel using Catmull-Rom basis functions
 * 
 * Implements high-quality bicubic interpolation using a 4x4 grid of samples.
 * Uses the standard bicubic kernel with parameter a = -0.5 (Catmull-Rom).
 * 
 * Advantages over bilinear:
 * - Smoother interpolation with C1 continuity
 * - Better preservation of high-frequency details
 * - Reduced aliasing artifacts
 * 
 * Trade-offs:
 * - Higher computational cost (16 samples vs 4)
 * - Requires boundary handling for edge cases
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct BicubicKernel : public InterpolationKernel<scalar_t, real_t> {
private:
    /**
     * Standard bicubic interpolation basis function with a = -0.5 (Catmull-Rom)
     * 
     * This is the classical cubic kernel used in image processing:
     * - Provides C1 continuity (smooth first derivatives)
     * - Support region: [-2, 2]
     * - Interpolates through control points (passes through data exactly)
     */
    inline real_t bicubic_kernel(real_t s) const {
        const real_t a = -0.5;  // Catmull-Rom parameter for optimal smoothness
        s = std::abs(s);  // Kernel is symmetric around 0
        
        if (s <= 1.0) {
            // Inner region: (a+2)|s|³ - (a+3)|s|² + 1
            // This region ensures interpolation (passes through control points)
            return (a + 2.0) * s * s * s - (a + 3.0) * s * s + 1.0;
        } else if (s <= 2.0) {
            // Outer region: a|s|³ - 5a|s|² + 8a|s| - 4a
            // This region provides smooth blending with neighboring samples
            return a * s * s * s - 5.0 * a * s * s + 8.0 * a * s - 4.0 * a;
        } else {
            // Beyond support region: no contribution
            return 0.0;
        }
    }
    
    /**
     * Derivative of the bicubic interpolation kernel
     * 
     * Required for computing gradients w.r.t. spatial coordinates.
     * Used in backpropagation for rotation parameter gradients.
     */
    inline real_t bicubic_kernel_derivative(real_t s) const {
        const real_t a = -0.5;
        real_t sign = (s < 0) ? -1.0 : 1.0;  // Preserve sign for derivative
        s = std::abs(s);
        
        if (s <= 1.0) {
            // Inner region derivative: 3(a+2)|s|² - 2(a+3)|s|
            return sign * (3.0 * (a + 2.0) * s * s - 2.0 * (a + 3.0) * s);
        } else if (s <= 2.0) {
            // Outer region derivative: 3a|s|² - 10a|s| + 8a
            return sign * (3.0 * a * s * s - 10.0 * a * s + 8.0 * a);
        } else {
            // Beyond support: no gradient
            return 0.0;
        }
    }
    
    /**
     * Safe sampling with edge clamping for out-of-bounds coordinates
     * 
     * Bicubic interpolation requires a 4x4 grid, which can extend beyond
     * the valid data region. This function handles boundary conditions by
     * clamping coordinates to the nearest valid sample.
     * 
     * This prevents artifacts that would occur from sampling undefined regions
     * while maintaining reasonable boundary behavior for the interpolation.
     */
    inline scalar_t sample_with_edge_clamping(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        int64_t r, int64_t c
    ) const {
        // Let sample_fftw_with_conjugate handle Friedel symmetry for c < 0
        // Only clamp if we're beyond the valid range after symmetry considerations
        
        // For c: after Friedel symmetry, clamp |c| to valid range [0, boxsize_half-1]
        if (std::abs(c) >= boxsize_half) {
            c = (c < 0) ? -(boxsize_half - 1) : (boxsize_half - 1);
        }
        
        // For r: clamp to valid range [-boxsize/2 + 1, boxsize/2]
        r = std::max(-boxsize / 2 + 1, std::min(r, boxsize / 2));
        
        return sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r, c);
    }

public:
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const override {
        // Extract integer and fractional parts
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);
        const real_t c_frac = c - c_floor;
        const real_t r_frac = r - r_floor;
        
        scalar_t result = scalar_t(0);
        
        // Sample 4x4 grid around the interpolation point
        // Grid extends from (r_floor-1, c_floor-1) to (r_floor+2, c_floor+2)
        for (int i = -1; i <= 2; ++i) {      // Row offset: covers 4 rows
            const real_t weight_r = bicubic_kernel(r_frac - i);

            for (int j = -1; j <= 2; ++j) {  // Column offset: covers 4 columns
                const scalar_t sample = sample_with_edge_clamping(rec, b, boxsize, boxsize_half, 
                                                     r_floor + i, c_floor + j);
                // Compute bicubic weights for this grid position
                const real_t weight_c = bicubic_kernel(c_frac - j);
                // Accumulate weighted contribution
                result += sample * weight_r * weight_c;
            }
        }
        
        return result;
    }
    
    std::tuple<scalar_t, scalar_t, scalar_t> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const override {
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);
        
        const real_t c_frac = c - c_floor;
        const real_t r_frac = r - r_floor;
        
        scalar_t val = scalar_t(0);     // Interpolated value
        scalar_t grad_r = scalar_t(0);  // Gradient w.r.t. row coordinate
        scalar_t grad_c = scalar_t(0);  // Gradient w.r.t. column coordinate
        
        // Sample 4x4 grid and compute value + gradients simultaneously
        // This is more efficient than separate passes
        for (int i = -1; i <= 2; ++i) {
            const real_t weight_r = bicubic_kernel(r_frac - i);
            const real_t dweight_r = bicubic_kernel_derivative(r_frac - i);
            
            for (int j = -1; j <= 2; ++j) {
                const scalar_t sample = sample_with_edge_clamping(rec, b, boxsize, boxsize_half, 
                                                     r_floor + i, c_floor + j);
                
                // Compute weights and their derivatives for this grid position
                const real_t weight_c = bicubic_kernel(c_frac - j);
                const real_t dweight_c = bicubic_kernel_derivative(c_frac - j);
                
                // Accumulate value and gradients
                val += sample * weight_r * weight_c;
                grad_r += sample * dweight_r * weight_c;  // Chain rule: ∂f/∂r
                grad_c += sample * weight_r * dweight_c;  // Chain rule: ∂f/∂c
            }
        }
        
        return std::make_tuple(val, grad_r, grad_c);
    }
};

/**
 * Factory function to create interpolation kernels
 * 
 * Returns the appropriate kernel implementation based on the interpolation method.
 * This allows the main projection functions to work with any interpolation type
 * through a common interface.
 * 
 * @param interpolation: String specifying method ("linear" or "cubic")
 * @return: Unique pointer to the appropriate kernel implementation
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
std::unique_ptr<InterpolationKernel<scalar_t, real_t>> get_interpolation_kernel(const std::string& interpolation) {
    if (interpolation == "linear") {
        return std::make_unique<BilinearKernel<scalar_t, real_t>>();
    } else if (interpolation == "cubic") {
        return std::make_unique<BicubicKernel<scalar_t, real_t>>();
    } else {
        throw std::runtime_error("Unsupported interpolation method: " + interpolation);
    }
}

/**
 * Abstract base class for backward projection (gradient distribution) kernels
 * 
 * These kernels implement the adjoint (transpose) operations of the forward
 * interpolation kernels. They distribute gradients from a single point to
 * the appropriate neighborhood in the reconstruction.
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct BackwardKernel {
    virtual ~BackwardKernel() = default;
    
    /**
     * Distribute gradient from a single point to reconstruction neighborhood
     * 
     * This is the adjoint operation of forward interpolation. The gradient
     * from one projection pixel is distributed to multiple reconstruction
     * voxels with appropriate weights.
     * 
     * @param accumulate_func: Function to accumulate gradients: f(r, c, grad_value)
     * @param grad_val: Gradient value to distribute
     * @param r: Row coordinate (floating point)
     * @param c: Column coordinate (floating point)
     */
    virtual void distribute_gradient(
        std::function<void(int64_t, int64_t, scalar_t)> accumulate_func,
        scalar_t grad_val,
        real_t r, real_t c
    ) const = 0;
};

/**
 * Bilinear backward kernel - distributes gradients to 2x2 neighborhood
 * 
 * This implements the adjoint of bilinear interpolation. Each gradient
 * is distributed to the 4 nearest neighbors with bilinear weights.
 * 
 * Mathematical relationship:
 * If forward: val = Σ w_i * data_i (sum over 4 neighbors)
 * Then backward: grad_data_i += w_i * grad_val (for each neighbor)
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct BilinearBackwardKernel : public BackwardKernel<scalar_t, real_t> {
    void distribute_gradient(
        std::function<void(int64_t, int64_t, scalar_t)> accumulate_func,
        scalar_t grad_val,
        real_t r, real_t c
    ) const override {
        // Extract integer and fractional parts
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);
        const real_t c_frac = c - c_floor;
        const real_t r_frac = r - r_floor;

        // Distribute gradient to 2x2 neighborhood with bilinear weights
        // These are exactly the same weights used in forward bilinear interpolation
        accumulate_func(r_floor,     c_floor,     grad_val * (1 - r_frac) * (1 - c_frac)); // Bottom-left
        accumulate_func(r_floor,     c_floor + 1, grad_val * (1 - r_frac) * c_frac);       // Bottom-right
        accumulate_func(r_floor + 1, c_floor,     grad_val * r_frac * (1 - c_frac));       // Top-left
        accumulate_func(r_floor + 1, c_floor + 1, grad_val * r_frac * c_frac);             // Top-right
    }
};

/**
 * Bicubic backward kernel - distributes gradients to 4x4 neighborhood
 * 
 * This implements the adjoint of bicubic interpolation. Each gradient
 * is distributed to a 4x4 grid of neighbors using the same bicubic weights
 * that were used in the forward pass.
 * 
 * This ensures mathematical consistency: the backward pass is the true
 * adjoint of the forward pass, which is critical for correct gradients
 * in optimization and machine learning applications.
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct BicubicBackwardKernel : public BackwardKernel<scalar_t, real_t> {
private:
    // Reuse the same bicubic kernel function from the forward implementation
    inline real_t bicubic_kernel(real_t s) const {
        const real_t a = -0.5;  // Catmull-Rom parameter
        s = std::abs(s);
        
        if (s <= 1.0) {
            return (a + 2.0) * s * s * s - (a + 3.0) * s * s + 1.0;
        } else if (s <= 2.0) {
            return a * s * s * s - 5.0 * a * s * s + 8.0 * a * s - 4.0 * a;
        } else {
            return 0.0;
        }
    }

public:
    void distribute_gradient(
        std::function<void(int64_t, int64_t, scalar_t)> accumulate_func,
        scalar_t grad_val,
        real_t r, real_t c
    ) const override {
        // Extract integer and fractional parts
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);
        const real_t c_frac = c - c_floor;
        const real_t r_frac = r - r_floor;
        
        // Distribute gradient to 4x4 neighborhood using bicubic weights
        // These are exactly the same weights used in forward bicubic interpolation
        for (int i = -1; i <= 2; ++i) {      // Row offset: covers 4 rows
            const real_t weight_r = bicubic_kernel(r_frac - i);

            for (int j = -1; j <= 2; ++j) {  // Column offset: covers 4 columns
                const real_t weight_c = bicubic_kernel(c_frac - j);
                const real_t total_weight = weight_r * weight_c;
                
                // Only distribute if weight is non-zero (bicubic has finite support)
                if (total_weight != 0.0) {
                    accumulate_func(r_floor + i, c_floor + j, grad_val * total_weight);
                }
            }
        }
    }
};

/**
 * Factory function to create backward projection kernels
 * 
 * Returns the appropriate backward kernel that implements the adjoint
 * of the corresponding forward interpolation method.
 * 
 * @param interpolation: String specifying method ("linear" or "cubic")
 * @return: Unique pointer to the appropriate backward kernel implementation
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
std::unique_ptr<BackwardKernel<scalar_t, real_t>> get_backward_kernel(const std::string& interpolation) {
    if (interpolation == "linear") {
        return std::make_unique<BilinearBackwardKernel<scalar_t, real_t>>();
    } else if (interpolation == "cubic") {
        return std::make_unique<BicubicBackwardKernel<scalar_t, real_t>>();
    } else {
        throw std::runtime_error("Unsupported interpolation method: " + interpolation);
    }
}



/**
 * Forward projection from 3D Fourier reconstruction to 2D projections
 * 
 * This is the main "gather" operation that samples a 3D Fourier-space reconstruction
 * at rotated coordinate grids to produce 2D projections. This implements the
 * Projection-Slice Theorem in Fourier space.
 * 
 * Algorithm:
 * 1. For each output pixel (i,j) in the projection
 * 2. Convert to Fourier coordinates (proj_coord_r, proj_coord_c)
 * 3. Apply rotation matrix to get sampling coordinates in reconstruction
 * 4. Apply oversampling scaling if specified
 * 5. Interpolate from reconstruction at these coordinates
 * 6. Apply phase shift if shifts are provided
 * 7. Store result in projection
 * 
 * @param reconstruction: 3D complex tensor [B, H, W/2+1] in FFTW format
 * @param rotations: 4D real tensor [B_rot, P, 2, 2] - 2x2 rotation matrices
 * @param shifts: Optional 3D real tensor [B_shift, P, 2] - translation shifts
 * @param output_shape: Shape of output projections [H_out, W_out]
 * @param interpolation: "linear" or "cubic"
 * @param oversampling: Scaling factor for coordinates (>1 for oversampling)
 * @param fourier_radius_cutoff: Optional frequency cutoff for low-pass filtering
 * @return: 4D complex tensor [B, P, H_out, W_out/2+1] - the projections
 */
at::Tensor forward_project_2d_cpu(
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef output_shape,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Input validation
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic", 
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(reconstruction.is_complex(), "Reconstruction must be a complex tensor");
    TORCH_CHECK(reconstruction.dim() == 3, "Reconstruction must be a 3D tensor (B, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2, "Rotations must be (B_rot, P, 2, 2)");

    // Extract tensor dimensions
    const auto B = reconstruction.size(0);          // Batch size
    const auto boxsize = reconstruction.size(1);    // Reconstruction height
    const auto boxsize_half = reconstruction.size(2); // Reconstruction width (FFTW format)
    
    const auto B_rot = rotations.size(0);           // Rotation batch size
    const auto P = rotations.size(1);               // Number of poses per batch
    TORCH_CHECK(B_rot == B || B_rot == 1, "Batch size of rotations must be 1 or same as reconstruction");

    const auto proj_boxsize = output_shape[0];      // Output projection size
    const auto proj_boxsize_half = output_shape[0] / 2 + 1; // Output width (FFTW format)
    
    // Initialize output tensor with zeros
    auto projection = torch::zeros({B, P, proj_boxsize, proj_boxsize_half}, reconstruction.options());

    // PyTorch dispatch macro for type-generic code
    // This allows the same code to work with float32 and float64
    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "forward_project_2d_cpu_rotations", ([&] {
        using rot_real_t = scalar_t;  // Real type for rotations/shifts
        auto rot_acc = rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        
        // Handle optional shifts parameter
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> shifts_acc;
        if (shifts.has_value()) {
            // Validate shifts tensor
            TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
            TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1, "Batch size of shifts must be 1 or same as reconstruction");
            TORCH_CHECK(shifts->size(1) == P, "Number of poses in shifts must match rotations");
            TORCH_CHECK(shifts->scalar_type() == rotations.scalar_type(), "Shifts and rotations must have the same dtype");
            shifts_acc.emplace(shifts->packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
        }
        
        // Dispatch for complex types (reconstruction data)
        AT_DISPATCH_COMPLEX_TYPES(reconstruction.scalar_type(), "forward_project_2d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;  // Extract real type from complex
            auto rec_acc = reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();
            auto proj_acc = projection.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();

            // Set up Fourier space filtering
            const real_t default_radius = proj_boxsize / 2.0;  // Nyquist frequency
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;  // Precompute for efficiency

            auto kernel = get_interpolation_kernel<scalar_t>(interpolation);

            // Main projection loop: parallelize over batch*pose combinations
            const int64_t grain_size = std::max(int64_t(1), (B * P) / (2 * at::get_num_threads()));
            at::parallel_for(0, B * P, grain_size, [&](int64_t start, int64_t end) {
                for (int64_t bp_idx = start; bp_idx < end; ++bp_idx) {
                    const int64_t b = bp_idx / P;           // Batch index
                    const int64_t p = bp_idx % P;           // Pose index
                    
                    // Handle broadcasting: same rotations for all batches if B_rot=1
                    const int64_t rot_b_idx = (B_rot == 1) ? 0 : b;
                    
                    for (int64_t i = 0; i < proj_boxsize; ++i) {        // Row (height)
                        for (int64_t j = 0; j < proj_boxsize_half; ++j) { // Column (width/2+1)
                            // Convert array indices to Fourier coordinates
                            // FFTW format: DC at (0,0), positive frequencies, then negative
                            real_t proj_coord_c = j;  // Column: always positive (FFTW half-space)
                            real_t proj_coord_r = (i <= proj_boxsize / 2) ? i : i - proj_boxsize; // Row: handle wrap-around

                            // Apply Fourier space filtering (low-pass)
                            if (proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r > radius_cutoff_sq) {
                                continue;  // Skip high frequencies
                            }
                            
                            // Apply oversampling scaling to coordinates
                            // Oversampling > 1 simulates zero-padding in real space
                            real_t sample_c = proj_coord_c * oversampling;
                            real_t sample_r = proj_coord_r * oversampling;
                            
                            // Apply rotation matrix to get sampling coordinates in reconstruction
                            // Matrix multiplication: [rot_c; rot_r] = R * [sample_c; sample_r]
                            real_t rot_c = rot_acc[rot_b_idx][p][0][0] * sample_c + rot_acc[rot_b_idx][p][0][1] * sample_r;
                            real_t rot_r = rot_acc[rot_b_idx][p][1][0] * sample_c + rot_acc[rot_b_idx][p][1][1] * sample_r;

                            // Interpolate from reconstruction at rotated coordinates
                            scalar_t val = kernel->interpolate(rec_acc, b, boxsize, boxsize_half, rot_r, rot_c);
                            
                            // Apply phase shift if translations are provided
                            // Shift in real space = phase modulation in Fourier space
                            if (shifts_acc.has_value()) {
                                const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                                // Compute phase: -2π * (k · shift)
                                real_t phase = -2.0 * M_PI * (proj_coord_r * (*shifts_acc)[shift_b_idx][p][0] / boxsize + 
                                                               proj_coord_c * (*shifts_acc)[shift_b_idx][p][1] / boxsize);
                                scalar_t phase_factor = scalar_t(cos(phase), sin(phase));  // e^(iφ)
                                val = val * phase_factor;  // Apply phase modulation
                            }
                            
                            proj_acc[b][p][i][j] = val;
                        }
                    }
                }
            });
        }));
    }));

    return projection;
}


/**
 * Unified backward projection function with smart gradient computation
 * 
 * This function computes gradients w.r.t. reconstruction, rotations, and shifts
 * based on what actually requires gradients. It automatically detects which
 * parameters need gradients using requires_grad() and only computes those,
 * avoiding unnecessary computation.
 * 
 * Features:
 * 1. Always computes reconstruction gradients (main scatter operation)
 * 2. Only computes rotation gradients if rotations.requires_grad() is true
 * 3. Only computes shift gradients if shifts exist and require gradients
 * 4. Uses proper adjoint operations for mathematical consistency
 * 
 * @return: Tuple of (grad_reconstruction, grad_rotations, grad_shifts)
 *          Empty tensors are returned for gradients that aren't needed
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_project_2d_cpu(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Input validation
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic", 
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(grad_projections.is_complex(), "Projections must be a complex tensor");
    TORCH_CHECK(grad_projections.dim() == 4, "Projections must be a 4D tensor (B, P, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2, "Rotations must be (B_rot, P, 2, 2)");

    const auto B = grad_projections.size(0);
    const auto P = grad_projections.size(1);
    const auto proj_boxsize = grad_projections.size(2);
    const auto proj_boxsize_half = grad_projections.size(3);
    
    const auto rec_boxsize = reconstruction.size(1);
    const auto rec_boxsize_half = reconstruction.size(2);
    
    // Always compute reconstruction gradients (this is the main scatter operation)
    auto grad_reconstruction = torch::zeros({B, rec_boxsize, rec_boxsize_half}, grad_projections.options());
    
    // Initialize gradient tensors based on what's actually needed
    at::Tensor grad_rotations;
    at::Tensor grad_shifts;
    
    const bool need_rotation_grads = rotations.requires_grad();
    const bool need_shift_grads = shifts.has_value() && shifts->requires_grad();
    
    if (need_rotation_grads) {
        grad_rotations = torch::zeros_like(rotations);
    } else {
        grad_rotations = torch::empty({0}, rotations.options());
    }
    
    if (need_shift_grads) {
        grad_shifts = torch::zeros_like(*shifts);
    } else {
        grad_shifts = torch::empty({0}, grad_projections.options().dtype(rotations.scalar_type()));
    }

    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "backward_project_2d_cpu", ([&] {
        using rot_real_t = scalar_t;
        auto rot_acc = rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        
        // Only create accessors for gradients that are actually needed
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 4, torch::DefaultPtrTraits>> grad_rot_acc;
        if (need_rotation_grads) {
            grad_rot_acc.emplace(grad_rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>());
        }
        
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> shifts_acc;
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> grad_shifts_acc;
        if (shifts.has_value()) {
            shifts_acc.emplace(shifts->packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
            if (need_shift_grads) {
                grad_shifts_acc.emplace(grad_shifts.packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
            }
        }

        AT_DISPATCH_COMPLEX_TYPES(grad_projections.scalar_type(), "backward_project_2d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;
            auto grad_proj_acc = grad_projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto grad_rec_acc = grad_reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();
            auto rec_acc = reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();

            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;

            auto kernel = get_interpolation_kernel<scalar_t>(interpolation);
            auto backward_kernel = get_backward_kernel<scalar_t>(interpolation);
            auto kernel_grad = get_interpolation_kernel<scalar_t>(interpolation);
            
            // Parallelize over batch*pose combinations
            const int64_t grain_size = std::max(int64_t(1), (B * P) / (2 * at::get_num_threads()));
            at::parallel_for(0, B * P, grain_size, [&](int64_t start, int64_t end) {
                for (int64_t bp_idx = start; bp_idx < end; ++bp_idx) {
                    const int64_t b = bp_idx / P;
                    const int64_t p = bp_idx % P;
                    
                    // Local accumulator arrays for this projection's gradients
                    rot_real_t local_rot_grad[2][2] = {{0, 0}, {0, 0}};      // 2x2 rotation matrix gradients
                    rot_real_t local_shift_grad[2] = {0, 0};                  // 2-element shift gradients
                    
                    // Lambda function to safely accumulate gradients with Friedel symmetry
                    auto accumulate_grad = [&](int64_t r, int64_t c, scalar_t grad) {
                        bool needs_conj = false;
                        
                        // Handle Friedel symmetry for negative column indices
                        if (c < 0) { 
                            c = -c;           // Mirror column to positive side
                            r = -r;           // Mirror row as well
                            needs_conj = true; // Need to conjugate the value
                        }
                        
                        // Bounds checking
                        if (c >= rec_boxsize_half) return;  // Beyond stored frequency range
                        if (r > rec_boxsize / 2 || r < -rec_boxsize / 2 + 1) return;  // Beyond valid row range

                        // Convert negative row indices to positive (FFTW wrapping)
                        int64_t r_eff = r < 0 ? rec_boxsize + r : r;
                        if (r_eff >= rec_boxsize) return;  // Final bounds check

                        // Atomically accumulate gradient (with conjugation if needed for Friedel symmetry)
                        scalar_t final_grad = needs_conj ? std::conj(grad) : grad;
                        atomic_add_complex(&grad_rec_acc[b][r_eff][c], final_grad);
                    };
                    const int64_t rot_b_idx = (rotations.size(0) == 1) ? 0 : b;
                    for (int64_t i = 0; i < proj_boxsize; ++i) {
                        for (int64_t j = 0; j < proj_boxsize_half; ++j) {
                            real_t proj_coord_c = j;
                            real_t proj_coord_r = (i <= proj_boxsize / 2) ? i : i - proj_boxsize;

                            if (proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r > radius_cutoff_sq) {
                                continue;
                            }

                            real_t sample_c = proj_coord_c * oversampling;
                            real_t sample_r = proj_coord_r * oversampling;
                            
                            real_t rot_c = rot_acc[rot_b_idx][p][0][0] * sample_c + rot_acc[rot_b_idx][p][0][1] * sample_r;
                            real_t rot_r = rot_acc[rot_b_idx][p][1][0] * sample_c + rot_acc[rot_b_idx][p][1][1] * sample_r;
                            
                            // Use abstracted interpolation kernel
                            scalar_t rec_val = kernel->interpolate(rec_acc, b, rec_boxsize, rec_boxsize_half, rot_r, rot_c);
                            scalar_t grad_proj = grad_proj_acc[b][p][i][j];

                            if (shifts_acc.has_value()) {
                               const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                               real_t phase = 2.0 * M_PI * (proj_coord_r * (*shifts_acc)[shift_b_idx][p][0] / rec_boxsize + proj_coord_c * (*shifts_acc)[shift_b_idx][p][1] / rec_boxsize);
                               scalar_t phase_factor = scalar_t(cos(phase), sin(phase));
                               grad_proj = grad_proj * phase_factor;
                            }

                            // Use abstracted backward kernel for gradient distribution
                            // This ensures the backward pass is the proper adjoint of the forward interpolation
                            backward_kernel->distribute_gradient(accumulate_grad, grad_proj, rot_r, rot_c);

                            // Compute gradients w.r.t. shift parameters (only if needed)
                            if (need_shift_grads) {
                                const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                                
                                // Apply phase modulation to reconstruction value for correct gradient
                                scalar_t modulated_rec_val = rec_val;
                                if (shifts_acc.has_value()) {
                                    real_t phase = -2.0 * M_PI * (proj_coord_r * (*shifts_acc)[shift_b_idx][p][0] / rec_boxsize + 
                                                                   proj_coord_c * (*shifts_acc)[shift_b_idx][p][1] / rec_boxsize);
                                    scalar_t phase_factor = scalar_t(cos(phase), sin(phase));
                                    modulated_rec_val = rec_val * phase_factor;
                                }
                                
                                // Compute phase derivatives: ∂φ/∂shift = -2π * coordinate / boxsize
                                scalar_t phase_grad_r = scalar_t(0, -2.0 * M_PI * proj_coord_r / rec_boxsize) * modulated_rec_val;
                                scalar_t phase_grad_c = scalar_t(0, -2.0 * M_PI * proj_coord_c / rec_boxsize) * modulated_rec_val;
                                
                                // Accumulate shift gradients locally (take real part of complex gradient)
                                scalar_t original_grad_proj = grad_proj_acc[b][p][i][j];
                                local_shift_grad[0] += (original_grad_proj * std::conj(phase_grad_r)).real();
                                local_shift_grad[1] += (original_grad_proj * std::conj(phase_grad_c)).real();
                            }

                            // Compute gradients w.r.t. rotation matrix elements (only if needed)
                            if (need_rotation_grads) {
                                auto [rec_val_unused, grad_r, grad_c] = kernel_grad->interpolate_with_gradients(rec_acc, b, rec_boxsize, rec_boxsize_half, rot_r, rot_c);
                                
                                // Chain rule: ∂f/∂R[i][j] = (∂f/∂rot_coord) * (∂rot_coord/∂R[i][j])
                                // 
                                // Rotation transformation: [rot_r; rot_c] = R * [sample_r; sample_c]
                                // Therefore:
                                // ∂rot_r/∂R[1][0] = sample_c,  ∂rot_r/∂R[1][1] = sample_r
                                // ∂rot_c/∂R[0][0] = sample_c,  ∂rot_c/∂R[0][1] = sample_r
                                //
                                // Accumulate gradients locally (taking real part for real-valued rotation matrices):
                                local_rot_grad[0][0] += (grad_proj * std::conj(grad_c * sample_c)).real();  // ∂f/∂R[0][0]
                                local_rot_grad[0][1] += (grad_proj * std::conj(grad_c * sample_r)).real();  // ∂f/∂R[0][1]
                                local_rot_grad[1][0] += (grad_proj * std::conj(grad_r * sample_c)).real();  // ∂f/∂R[1][0]
                                local_rot_grad[1][1] += (grad_proj * std::conj(grad_r * sample_r)).real();  // ∂f/∂R[1][1]
                            }
                        }
                    }
                    
                    // Atomically add local gradients to global tensors
                    if (need_rotation_grads && grad_rot_acc.has_value()) {
                        const int64_t rot_b_idx = (rotations.size(0) == 1) ? 0 : b;
                        atomic_add_real(&(*grad_rot_acc)[rot_b_idx][p][0][0], local_rot_grad[0][0]);
                        atomic_add_real(&(*grad_rot_acc)[rot_b_idx][p][0][1], local_rot_grad[0][1]);
                        atomic_add_real(&(*grad_rot_acc)[rot_b_idx][p][1][0], local_rot_grad[1][0]);
                        atomic_add_real(&(*grad_rot_acc)[rot_b_idx][p][1][1], local_rot_grad[1][1]);
                    }
                    
                    if (need_shift_grads && grad_shifts_acc.has_value()) {
                        const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                        atomic_add_real(&(*grad_shifts_acc)[shift_b_idx][p][0], local_shift_grad[0]);
                        atomic_add_real(&(*grad_shifts_acc)[shift_b_idx][p][1], local_shift_grad[1]);
                    }
                }
            });
        }));
    }));

    return std::make_tuple(grad_reconstruction, grad_rotations, grad_shifts);
} 