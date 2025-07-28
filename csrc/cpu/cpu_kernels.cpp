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

#include "cpu_kernels.h"
#include <torch/extension.h>
#include <complex>
#include <algorithm>

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
            for (int j = -1; j <= 2; ++j) {  // Column offset: covers 4 columns
                const scalar_t sample = sample_with_edge_clamping(rec, b, boxsize, boxsize_half, 
                                                     r_floor + i, c_floor + j);
                // Compute bicubic weights for this grid position
                const real_t weight_r = bicubic_kernel(r_frac - i);
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
            for (int j = -1; j <= 2; ++j) {
                const scalar_t sample = sample_with_edge_clamping(rec, b, boxsize, boxsize_half, 
                                                     r_floor + i, c_floor + j);
                
                // Compute weights and their derivatives for this grid position
                const real_t weight_r = bicubic_kernel(r_frac - i);
                const real_t weight_c = bicubic_kernel(c_frac - j);
                const real_t dweight_r = bicubic_kernel_derivative(r_frac - i);
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

            // Main projection loop: iterate over all output pixels
            for (int64_t b = 0; b < B; ++b) {           // Batch dimension
                for (int64_t p = 0; p < P; ++p) {       // Pose dimension
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
                            auto kernel = get_interpolation_kernel<scalar_t>(interpolation);
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
            }
        }));
    }));

    return projection;
}

/**
 * Backward projection from 2D projections to 3D Fourier reconstruction
 * 
 * This is the main "scatter" operation that takes gradients from 2D projections
 * and accumulates them into a 3D reconstruction. This is the adjoint (transpose)
 * of the forward projection operation.
 * 
 * Algorithm:
 * 1. For each projection pixel with gradient grad_val
 * 2. Convert to Fourier coordinates and apply rotation
 * 3. For bilinear: distribute grad_val to 4 nearest neighbors with bilinear weights
 * 4. For bicubic: would distribute to 16 neighbors (currently uses bilinear fallback)
 * 5. Handle Friedel symmetry when accumulating gradients
 * 
 * Note: This function currently only implements bilinear backward projection,
 * even when cubic interpolation is requested. This is a limitation that should
 * be addressed for full bicubic support.
 * 
 * @param grad_projections: 4D complex tensor [B, P, H, W/2+1] - projection gradients
 * @param rotations: 4D real tensor [B_rot, P, 2, 2] - rotation matrices
 * @param shifts: Optional shifts (used for phase correction)
 * @param reconstruction_shape: Shape of output reconstruction [H, W/2+1]
 * @param interpolation: Interpolation method (currently only affects validation)
 * @param oversampling: Coordinate scaling factor
 * @return: 3D complex tensor [B, H, W/2+1] - accumulated reconstruction gradients
 */
at::Tensor backward_project_2d_cpu(
    const at::Tensor& grad_projections,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef reconstruction_shape,
    const std::string& interpolation,
    const double oversampling
) {
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic", 
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(grad_projections.is_complex(), "Projections must be a complex tensor");
    TORCH_CHECK(grad_projections.dim() == 4, "Projections must be a 4D tensor (B, P, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2, "Rotations must be (B_rot, P, 2, 2)");

    const auto B = grad_projections.size(0);
    const auto P = grad_projections.size(1);
    const auto proj_boxsize = grad_projections.size(2);
    const auto proj_boxsize_half = grad_projections.size(3);
    
    const auto boxsize = reconstruction_shape[0];
    const auto boxsize_half = reconstruction_shape[1];

    auto grad_reconstruction = torch::zeros({B, boxsize, boxsize_half}, grad_projections.options());

    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "backward_project_2d_cpu_rotations", ([&] {
        using rot_real_t = scalar_t;
        auto rot_acc = rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> shifts_acc;
        if (shifts.has_value()) {
            shifts_acc.emplace(shifts->packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
        }
        
        AT_DISPATCH_COMPLEX_TYPES(grad_projections.scalar_type(), "backward_project_2d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;
            auto grad_rec_acc = grad_reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();
            auto grad_proj_acc = grad_projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();

            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff_sq = default_radius * default_radius;
            
            for (int64_t b = 0; b < B; ++b) {
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
                    if (c >= boxsize_half) return;  // Beyond stored frequency range
                    if (r > boxsize / 2 || r < -boxsize / 2 + 1) return;  // Beyond valid row range

                    // Convert negative row indices to positive (FFTW wrapping)
                    int64_t r_eff = r < 0 ? boxsize + r : r;
                    if (r_eff >= boxsize) return;  // Final bounds check

                    // Accumulate gradient (with conjugation if needed for Friedel symmetry)
                    grad_rec_acc[b][r_eff][c] += needs_conj ? std::conj(grad) : grad;
                };

                for (int64_t p = 0; p < P; ++p) {
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

                            scalar_t grad_val = grad_proj_acc[b][p][i][j];
                            
                            // Apply phase correction for shifts (adjoint of forward phase modulation)
                            if (shifts_acc.has_value()) {
                                const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                                // Note: opposite sign compared to forward projection
                                real_t phase = 2.0 * M_PI * (proj_coord_r * (*shifts_acc)[shift_b_idx][p][0] / boxsize + 
                                                              proj_coord_c * (*shifts_acc)[shift_b_idx][p][1] / boxsize);
                                scalar_t phase_factor = scalar_t(cos(phase), sin(phase));
                                grad_val = grad_val * phase_factor;
                            }

                            // Bilinear distribution of gradients to 4 nearest neighbors
                            // This implements the adjoint of bilinear interpolation
                            const int64_t c_floor = floor(rot_c);
                            const int64_t r_floor = floor(rot_r);
                            const real_t c_frac = rot_c - c_floor;
                            const real_t r_frac = rot_r - r_floor;

                            // Distribute gradient with bilinear weights (adjoint operation)
                            accumulate_grad(r_floor, c_floor, grad_val * (1 - r_frac) * (1 - c_frac));        // Bottom-left
                            accumulate_grad(r_floor, c_floor + 1, grad_val * (1 - r_frac) * c_frac);          // Bottom-right
                            accumulate_grad(r_floor + 1, c_floor, grad_val * r_frac * (1 - c_frac));          // Top-left
                            accumulate_grad(r_floor + 1, c_floor + 1, grad_val * r_frac * c_frac);            // Top-right
                        }
                    }
                }
            }
        }));
    }));

    return grad_reconstruction;
} 

/**
 * Compute gradients w.r.t. reconstruction, rotations, and shifts
 * 
 * This function computes the full set of gradients needed for backpropagation
 * through the forward projection operation. It combines:
 * 1. Gradient w.r.t. reconstruction (via backward_project_2d_cpu)
 * 2. Gradients w.r.t. rotation matrices (via chain rule with interpolation gradients)
 * 3. Gradients w.r.t. shift parameters (via phase derivative)
 * 
 * The rotation gradients use analytical derivatives of the interpolation kernels,
 * which is more accurate and efficient than finite differences.
 * 
 * @return: Tuple of (grad_reconstruction, grad_rotations, grad_shifts)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_project_2d_cpu_adj(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Compute gradient w.r.t. reconstruction using standard backward projection
    auto grad_reconstruction = backward_project_2d_cpu(
        grad_projections,
        rotations,
        shifts,
        {reconstruction.size(1), reconstruction.size(2)},
        interpolation,
        oversampling
    );

    const auto B = grad_projections.size(0);
    const auto P = grad_projections.size(1);
    const auto proj_boxsize = grad_projections.size(2);
    const auto proj_boxsize_half = grad_projections.size(3);
    
    const auto rec_boxsize = reconstruction.size(1);
    const auto rec_boxsize_half = reconstruction.size(2);
    
    // Initialize gradient tensors for rotation matrices and shifts
    auto grad_rotations = torch::zeros_like(rotations);
    auto grad_shifts = shifts.has_value() ? torch::zeros_like(*shifts) : torch::empty({0});

    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "backward_project_2d_cpu_adj_rotations", ([&] {
        using rot_real_t = scalar_t;
        auto rot_acc = rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        auto grad_rot_acc = grad_rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> shifts_acc;
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> grad_shifts_acc;
        if (shifts.has_value()) {
            shifts_acc.emplace(shifts->packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
            grad_shifts_acc.emplace(grad_shifts.packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
        }

        AT_DISPATCH_COMPLEX_TYPES(grad_projections.scalar_type(), "backward_project_2d_cpu_adj", ([&] {
            using real_t = typename scalar_t::value_type;
            auto grad_proj_acc = grad_projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto rec_acc = reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();

            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;

            for (int64_t b = 0; b < B; ++b) {
                for (int64_t p = 0; p < P; ++p) {
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
                            auto kernel = get_interpolation_kernel<scalar_t>(interpolation);
                            scalar_t rec_val = kernel->interpolate(rec_acc, b, rec_boxsize, rec_boxsize_half, rot_r, rot_c);
                            scalar_t grad_proj = grad_proj_acc[b][p][i][j];

                            if (shifts_acc.has_value()) {
                               const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                               real_t phase = 2.0 * M_PI * (proj_coord_r * (*shifts_acc)[shift_b_idx][p][0] / rec_boxsize + proj_coord_c * (*shifts_acc)[shift_b_idx][p][1] / rec_boxsize);
                               scalar_t phase_factor = scalar_t(cos(phase), sin(phase));
                               grad_proj = grad_proj * phase_factor;
                            }

                            // Compute gradients w.r.t. shift parameters
                            if (grad_shifts_acc.has_value()) {
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
                                
                                // Accumulate shift gradients (take real part of complex gradient)
                                scalar_t original_grad_proj = grad_proj_acc[b][p][i][j];
                                (*grad_shifts_acc)[shift_b_idx][p][0] += (original_grad_proj * std::conj(phase_grad_r)).real();
                                (*grad_shifts_acc)[shift_b_idx][p][1] += (original_grad_proj * std::conj(phase_grad_c)).real();
                            }

                            // Compute gradients w.r.t. rotation matrix elements using chain rule
                            auto kernel_grad = get_interpolation_kernel<scalar_t>(interpolation);
                            auto [rec_val_unused, grad_r, grad_c] = kernel_grad->interpolate_with_gradients(rec_acc, b, rec_boxsize, rec_boxsize_half, rot_r, rot_c);
                            
                            // Chain rule: ∂f/∂R[i][j] = (∂f/∂rot_coord) * (∂rot_coord/∂R[i][j])
                            // 
                            // Rotation transformation: [rot_r; rot_c] = R * [sample_r; sample_c]
                            // Therefore:
                            // ∂rot_r/∂R[1][0] = sample_c,  ∂rot_r/∂R[1][1] = sample_r
                            // ∂rot_c/∂R[0][0] = sample_c,  ∂rot_c/∂R[0][1] = sample_r
                            //
                            // Final gradients (taking real part for real-valued rotation matrices):
                            grad_rot_acc[rot_b_idx][p][0][0] += (grad_proj * std::conj(grad_c * sample_c)).real();  // ∂f/∂R[0][0]
                            grad_rot_acc[rot_b_idx][p][0][1] += (grad_proj * std::conj(grad_c * sample_r)).real();  // ∂f/∂R[0][1]
                            grad_rot_acc[rot_b_idx][p][1][0] += (grad_proj * std::conj(grad_r * sample_c)).real();  // ∂f/∂R[1][0]
                            grad_rot_acc[rot_b_idx][p][1][1] += (grad_proj * std::conj(grad_r * sample_r)).real();  // ∂f/∂R[1][1]
                        }
                    }
                }
            }
        }));
    }));

    return std::make_tuple(grad_reconstruction, grad_rotations, grad_shifts);
} 