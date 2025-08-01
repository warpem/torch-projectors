/**
 * CPU Kernels for Differentiable 3D->2D Projection Operations in Fourier Space
 * 
 * This file implements high-performance CPU kernels for forward and backward projection
 * operations that project 3D Fourier volumes to 2D projections. Used in cryo-electron
 * tomography and related 3D imaging applications.
 * 
 * Key Concepts:
 * - Forward projection: Sample from 4D Fourier reconstruction to create 2D projections
 * - Backward projection: Scatter 2D projection gradients into 4D Fourier reconstruction
 * - Central slice theorem: 2D projection corresponds to 3D central slice through origin
 * - 3D Friedel symmetry: F(kx,ky,kz) = conj(F(-kx,-ky,-kz)) for real-valued reconstructions
 * - Interpolation: Trilinear (8 neighbors) and tricubic (64 neighbors) with analytical gradients
 */

#include "projection_3d_to_2d_kernels.h"
#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <complex>
#include <algorithm>
#include <atomic>
#include <omp.h>

/**
 * Atomic accumulation for complex numbers using separate real/imaginary parts
 * (Reused from 2D implementation for consistency)
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline void atomic_add_complex(scalar_t* target, const scalar_t& value) {
    std::atomic<real_t>* real_ptr = reinterpret_cast<std::atomic<real_t>*>(target);
    std::atomic<real_t>* imag_ptr = real_ptr + 1;
    
    real_ptr->fetch_add(value.real(), std::memory_order_relaxed);
    imag_ptr->fetch_add(value.imag(), std::memory_order_relaxed);
}

/**
 * Atomic accumulation for real numbers
 * (Reused from 2D implementation for consistency)
 */
template <typename real_t>
inline void atomic_add_real(real_t* target, const real_t& value) {
    std::atomic<real_t>* atomic_ptr = reinterpret_cast<std::atomic<real_t>*>(target);
    atomic_ptr->fetch_add(value, std::memory_order_relaxed);
}

/**
 * Sample from 3D FFTW-formatted Fourier space with automatic Friedel symmetry handling
 * 
 * Extends the 2D FFTW sampling to 3D volumes. FFTW real-to-complex format stores only 
 * positive frequencies in the last dimension. For negative frequencies, we use 3D Friedel 
 * symmetry: F(-kx,-ky,-kz) = conj(F(kx,ky,kz))
 * 
 * @param rec: 4D complex tensor [batch, depth, height, width/2+1] in FFTW format
 * @param b: batch index
 * @param boxsize: full size of the reconstruction (width before RFFT)
 * @param boxsize_half: width/2+1 (actual stored width)
 * @param d: depth coordinate (can be negative, handled via wrapping)
 * @param r: row coordinate (can be negative, handled via wrapping) 
 * @param c: column coordinate (can be negative, handled via Friedel symmetry)
 * @return: Complex value at (d,r,c) with proper symmetry handling
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline scalar_t sample_3d_fftw_with_conjugate(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
    const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
    int64_t d, int64_t r, int64_t c
) {
    bool need_conjugate = false;

    // Handle negative kx via 3D Friedel symmetry (c < 0)
    // For real-valued reconstructions: F(-kx,-ky,-kz) = conj(F(kx,ky,kz))
    if (c < 0) {
        c = -c;          // Mirror to positive kx
        r = -r;          // ky must be mirrored as well
        d = -d;          // kz must be mirrored as well  
        need_conjugate = !need_conjugate;
    }

    // Clamp coordinates to valid array bounds
    c = std::min(c, boxsize_half - 1);  // Column: [0, boxsize/2]
    
    // Row and depth: [-boxsize/2+1, boxsize/2]
    r = std::min(boxsize / 2, std::max(r, -boxsize / 2 + 1));
    d = std::min(boxsize / 2, std::max(d, -boxsize / 2 + 1));

    // Convert coordinate system to array indices (FFTW wrapping)
    // In FFTW format, coordinate 0 corresponds to the center of the array
    // Negative frequencies are stored at the end of the array
    if (r < 0) r = boxsize + r;
    if (d < 0) d = boxsize + d;

    // Final bounds check
    r = std::min(r, boxsize - 1);
    d = std::min(d, boxsize - 1);

    // Return conjugated value if we used Friedel symmetry
    if (need_conjugate)
        return std::conj(rec[b][d][r][c]);
    else
        return rec[b][d][r][c];
}

/**
 * Interpolation method enumeration for 3D operations
 */
enum class Interpolation3DMethod {
    LINEAR,  // Trilinear interpolation (2x2x2 = 8 grid points)
    CUBIC    // Tricubic interpolation (4x4x4 = 64 grid points)
};

/**
 * Abstract base class for 3D interpolation kernels
 * 
 * Provides a unified interface for different 3D interpolation methods.
 * Each kernel must implement both forward interpolation and gradient computation.
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct Interpolation3DKernel {
    virtual ~Interpolation3DKernel() = default;
    
    /**
     * Perform 3D interpolation at continuous coordinates (d,r,c)
     * @param rec: Source 4D tensor to sample from
     * @param b: Batch index
     * @param boxsize: Full reconstruction size
     * @param boxsize_half: Half size (for FFTW format)
     * @param d: Depth coordinate (floating point)
     * @param r: Row coordinate (floating point)
     * @param c: Column coordinate (floating point)
     * @return: Interpolated complex value
     */
    virtual scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t d, real_t r, real_t c
    ) const = 0;
    
    /**
     * Perform 3D interpolation with simultaneous gradient computation
     * Used for computing gradients w.r.t. rotation parameters
     * @return: Tuple of (value, grad_d, grad_r, grad_c)
     */
    virtual std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t d, real_t r, real_t c
    ) const = 0;
};

/**
 * Trilinear interpolation kernel
 * 
 * Samples a 2x2x2 = 8 grid of neighboring voxels and performs trilinear interpolation.
 * This is the 3D extension of bilinear interpolation.
 * 
 * Mathematical formula:
 * f(x,y,z) = Σ f(i,j,k) * (1-|x-i|) * (1-|y-j|) * (1-|z-k|)
 * where the sum is over the 8 corner points of the unit cube.
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct TrilinearKernel : public Interpolation3DKernel<scalar_t, real_t> {
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t d, real_t r, real_t c
    ) const override {
        // Extract integer and fractional parts of coordinates
        const int64_t d_floor = floor(d);
        const int64_t r_floor = floor(r);
        const int64_t c_floor = floor(c);
        const real_t d_frac = d - d_floor;  // Fractional part [0,1)
        const real_t r_frac = r - r_floor;  // Fractional part [0,1)
        const real_t c_frac = c - c_floor;  // Fractional part [0,1)

        // Sample 2x2x2 = 8 neighboring voxels
        // Using systematic naming: pDRC where D,R,C ∈ {0,1} indicate the offset
        const scalar_t p000 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor,     r_floor,     c_floor);
        const scalar_t p001 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor,     r_floor,     c_floor + 1);
        const scalar_t p010 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor,     r_floor + 1, c_floor);
        const scalar_t p011 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor,     r_floor + 1, c_floor + 1);
        const scalar_t p100 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor,     c_floor);
        const scalar_t p101 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor,     c_floor + 1);
        const scalar_t p110 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor + 1, c_floor);
        const scalar_t p111 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor + 1, c_floor + 1);

        // Trilinear interpolation: interpolate in each dimension sequentially
        // First, interpolate along the c dimension (4 edge interpolations)
        const scalar_t p00 = p000 + (p001 - p000) * c_frac;  // Back-bottom edge
        const scalar_t p01 = p010 + (p011 - p010) * c_frac;  // Back-top edge  
        const scalar_t p10 = p100 + (p101 - p100) * c_frac;  // Front-bottom edge
        const scalar_t p11 = p110 + (p111 - p110) * c_frac;  // Front-top edge

        // Second, interpolate along the r dimension (2 face interpolations)
        const scalar_t p0 = p00 + (p01 - p00) * r_frac;  // Back face
        const scalar_t p1 = p10 + (p11 - p10) * r_frac;  // Front face

        // Finally, interpolate along the d dimension (final result)
        return p0 + (p1 - p0) * d_frac;
    }
    
    std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t d, real_t r, real_t c
    ) const override {
        const int64_t d_floor = floor(d);
        const int64_t r_floor = floor(r);
        const int64_t c_floor = floor(c);
        
        const real_t d_frac = d - d_floor;
        const real_t r_frac = r - r_floor;
        const real_t c_frac = c - c_floor;

        // Sample 2x2x2 grid
        const scalar_t p000 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor,     r_floor,     c_floor);
        const scalar_t p001 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor,     r_floor,     c_floor + 1);
        const scalar_t p010 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor,     r_floor + 1, c_floor);
        const scalar_t p011 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor,     r_floor + 1, c_floor + 1);
        const scalar_t p100 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor,     c_floor);
        const scalar_t p101 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor,     c_floor + 1);
        const scalar_t p110 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor + 1, c_floor);
        const scalar_t p111 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor + 1, c_floor + 1);

        // Value computation (same as interpolate method)
        const scalar_t p00 = p000 + (p001 - p000) * c_frac;
        const scalar_t p01 = p010 + (p011 - p010) * c_frac;
        const scalar_t p10 = p100 + (p101 - p100) * c_frac;
        const scalar_t p11 = p110 + (p111 - p110) * c_frac;
        
        const scalar_t p0 = p00 + (p01 - p00) * r_frac;
        const scalar_t p1 = p10 + (p11 - p10) * r_frac;
        
        const scalar_t val = p0 + (p1 - p0) * d_frac;

        // Analytical spatial gradients derived from trilinear formula
        // ∂f/∂d, ∂f/∂r, ∂f/∂c computed analytically for efficiency
        const scalar_t grad_d = p1 - p0;  // ∂f/∂d
        const scalar_t grad_r = (1 - d_frac) * (p01 - p00) + d_frac * (p11 - p10);  // ∂f/∂r
        const scalar_t grad_c = (1 - d_frac) * (1 - r_frac) * (p001 - p000) +
                                (1 - d_frac) * r_frac * (p011 - p010) +
                                d_frac * (1 - r_frac) * (p101 - p100) +
                                d_frac * r_frac * (p111 - p110);  // ∂f/∂c

        return std::make_tuple(val, grad_d, grad_r, grad_c);
    }
};

/**
 * Tricubic interpolation kernel using separable 1D cubic kernels
 * 
 * Implements high-quality tricubic interpolation using a 4x4x4 = 64 grid of samples.
 * Uses the standard bicubic kernel with parameter a = -0.5 (Catmull-Rom) applied
 * separably in each dimension.
 * 
 * Advantages over trilinear:
 * - Smoother interpolation with C1 continuity in all dimensions
 * - Better preservation of high-frequency details
 * - Reduced aliasing artifacts
 * 
 * Trade-offs:
 * - Much higher computational cost (64 samples vs 8)
 * - Requires more sophisticated boundary handling
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct TricubicKernel : public Interpolation3DKernel<scalar_t, real_t> {
private:
    /**
     * Standard bicubic interpolation basis function with a = -0.5 (Catmull-Rom) 
     * (Reused from 2D implementation for consistency)
     */
    inline real_t cubic_kernel(real_t s) const {
        const real_t a = -0.5;  // Catmull-Rom parameter
        s = std::abs(s);  // Kernel is symmetric around 0
        
        if (s <= 1.0) {
            return (a + 2.0) * s * s * s - (a + 3.0) * s * s + 1.0;
        } else if (s <= 2.0) {
            return a * s * s * s - 5.0 * a * s * s + 8.0 * a * s - 4.0 * a;
        } else {
            return 0.0;
        }
    }
    
    /**
     * Derivative of the bicubic interpolation kernel
     * (Reused from 2D implementation for consistency)
     */
    inline real_t cubic_kernel_derivative(real_t s) const {
        const real_t a = -0.5;
        real_t sign = (s < 0) ? -1.0 : 1.0;
        s = std::abs(s);
        
        if (s <= 1.0) {
            return sign * (3.0 * (a + 2.0) * s * s - 2.0 * (a + 3.0) * s);
        } else if (s <= 2.0) {
            return sign * (3.0 * a * s * s - 10.0 * a * s + 8.0 * a);
        } else {
            return 0.0;
        }
    }
    
    /**
     * Safe 3D sampling with edge clamping for out-of-bounds coordinates
     * 
     * Tricubic interpolation requires a 4x4x4 grid, which can extend beyond
     * the valid data region. This function handles boundary conditions by
     * clamping coordinates to the nearest valid sample.
     */
    inline scalar_t sample_with_edge_clamping_3d(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        int64_t d, int64_t r, int64_t c
    ) const {
        // For c: after Friedel symmetry, clamp |c| to valid range [0, boxsize_half-1]
        if (std::abs(c) >= boxsize_half) {
            c = (c < 0) ? -(boxsize_half - 1) : (boxsize_half - 1);
        }
        
        // For r and d: clamp to valid range [-boxsize/2 + 1, boxsize/2]
        r = std::max(-boxsize / 2 + 1, std::min(r, boxsize / 2));
        d = std::max(-boxsize / 2 + 1, std::min(d, boxsize / 2));
        
        return sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, d, r, c);
    }

public:
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t d, real_t r, real_t c
    ) const override {
        // Extract integer and fractional parts
        const int64_t d_floor = floor(d);
        const int64_t r_floor = floor(r);
        const int64_t c_floor = floor(c);
        const real_t d_frac = d - d_floor;
        const real_t r_frac = r - r_floor;
        const real_t c_frac = c - c_floor;
        
        scalar_t result = scalar_t(0);
        
        // Sample 4x4x4 = 64 grid around the interpolation point
        // Grid extends from (d_floor-1, r_floor-1, c_floor-1) to (d_floor+2, r_floor+2, c_floor+2)
        for (int k = -1; k <= 2; ++k) {      // Depth offset: covers 4 depths
            const real_t weight_d = cubic_kernel(d_frac - k);
            
            for (int i = -1; i <= 2; ++i) {  // Row offset: covers 4 rows
                const real_t weight_r = cubic_kernel(r_frac - i);

                for (int j = -1; j <= 2; ++j) {  // Column offset: covers 4 columns
                    const scalar_t sample = sample_with_edge_clamping_3d(rec, b, boxsize, boxsize_half,
                                                                        d_floor + k, r_floor + i, c_floor + j);
                    // Compute tricubic weights for this grid position (separable)
                    const real_t weight_c = cubic_kernel(c_frac - j);
                    // Accumulate weighted contribution
                    result += sample * weight_d * weight_r * weight_c;
                }
            }
        }
        
        return result;
    }
    
    std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t d, real_t r, real_t c
    ) const override {
        const int64_t d_floor = floor(d);
        const int64_t r_floor = floor(r);
        const int64_t c_floor = floor(c);
        
        const real_t d_frac = d - d_floor;
        const real_t r_frac = r - r_floor;
        const real_t c_frac = c - c_floor;
        
        scalar_t val = scalar_t(0);       // Interpolated value
        scalar_t grad_d = scalar_t(0);    // Gradient w.r.t. depth coordinate
        scalar_t grad_r = scalar_t(0);    // Gradient w.r.t. row coordinate
        scalar_t grad_c = scalar_t(0);    // Gradient w.r.t. column coordinate
        
        // Sample 4x4x4 grid and compute value + gradients simultaneously
        for (int k = -1; k <= 2; ++k) {
            const real_t weight_d = cubic_kernel(d_frac - k);
            const real_t dweight_d = cubic_kernel_derivative(d_frac - k);
            
            for (int i = -1; i <= 2; ++i) {
                const real_t weight_r = cubic_kernel(r_frac - i);
                const real_t dweight_r = cubic_kernel_derivative(r_frac - i);
                
                for (int j = -1; j <= 2; ++j) {
                    const scalar_t sample = sample_with_edge_clamping_3d(rec, b, boxsize, boxsize_half,
                                                                        d_floor + k, r_floor + i, c_floor + j);
                    
                    // Compute weights and their derivatives for this grid position
                    const real_t weight_c = cubic_kernel(c_frac - j);
                    const real_t dweight_c = cubic_kernel_derivative(c_frac - j);
                    
                    // Accumulate value and gradients (separable cubic kernels)
                    val += sample * weight_d * weight_r * weight_c;
                    grad_d += sample * dweight_d * weight_r * weight_c;  // Chain rule: ∂f/∂d
                    grad_r += sample * weight_d * dweight_r * weight_c;  // Chain rule: ∂f/∂r
                    grad_c += sample * weight_d * weight_r * dweight_c;  // Chain rule: ∂f/∂c
                }
            }
        }
        
        return std::make_tuple(val, grad_d, grad_r, grad_c);
    }
};

/**
 * Factory function to create 3D interpolation kernels
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
std::unique_ptr<Interpolation3DKernel<scalar_t, real_t>> get_3d_interpolation_kernel(const std::string& interpolation) {
    if (interpolation == "linear") {
        return std::make_unique<TrilinearKernel<scalar_t, real_t>>();
    } else if (interpolation == "cubic") {
        return std::make_unique<TricubicKernel<scalar_t, real_t>>();
    } else {
        throw std::runtime_error("Unsupported 3D interpolation method: " + interpolation);
    }
}

/**
 * Abstract base class for backward projection (gradient distribution) kernels for 3D->2D
 * 
 * These kernels implement the adjoint (transpose) operations of the forward
 * 3D interpolation kernels. They distribute gradients from a single 2D projection
 * point to the appropriate 3D neighborhood in the reconstruction.
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct Backward3DKernel {
    virtual ~Backward3DKernel() = default;
    
    /**
     * Distribute gradient from a single 2D projection point to 3D reconstruction neighborhood
     * 
     * This is the adjoint operation of forward 3D interpolation. The gradient
     * from one projection pixel is distributed to multiple reconstruction
     * voxels with appropriate weights.
     * 
     * @param accumulate_func: Function to accumulate gradients: f(d, r, c, grad_value)
     * @param grad_val: Gradient value to distribute
     * @param d: Depth coordinate (floating point)
     * @param r: Row coordinate (floating point)
     * @param c: Column coordinate (floating point)
     */
    virtual void distribute_gradient(
        std::function<void(int64_t, int64_t, int64_t, scalar_t)> accumulate_func,
        scalar_t grad_val,
        real_t d, real_t r, real_t c
    ) const = 0;
};

/**
 * Trilinear backward kernel - distributes gradients to 2x2x2 neighborhood
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct TrilinearBackwardKernel : public Backward3DKernel<scalar_t, real_t> {
    void distribute_gradient(
        std::function<void(int64_t, int64_t, int64_t, scalar_t)> accumulate_func,
        scalar_t grad_val,
        real_t d, real_t r, real_t c
    ) const override {
        // Extract integer and fractional parts
        const int64_t d_floor = floor(d);
        const int64_t r_floor = floor(r);
        const int64_t c_floor = floor(c);
        const real_t d_frac = d - d_floor;
        const real_t r_frac = r - r_floor;
        const real_t c_frac = c - c_floor;

        // Distribute gradient to 2x2x2 neighborhood with trilinear weights
        // These are exactly the same weights used in forward trilinear interpolation
        accumulate_func(d_floor,     r_floor,     c_floor,     grad_val * (1 - d_frac) * (1 - r_frac) * (1 - c_frac)); // p000
        accumulate_func(d_floor,     r_floor,     c_floor + 1, grad_val * (1 - d_frac) * (1 - r_frac) * c_frac);       // p001
        accumulate_func(d_floor,     r_floor + 1, c_floor,     grad_val * (1 - d_frac) * r_frac * (1 - c_frac));       // p010
        accumulate_func(d_floor,     r_floor + 1, c_floor + 1, grad_val * (1 - d_frac) * r_frac * c_frac);             // p011
        accumulate_func(d_floor + 1, r_floor,     c_floor,     grad_val * d_frac * (1 - r_frac) * (1 - c_frac));       // p100
        accumulate_func(d_floor + 1, r_floor,     c_floor + 1, grad_val * d_frac * (1 - r_frac) * c_frac);             // p101
        accumulate_func(d_floor + 1, r_floor + 1, c_floor,     grad_val * d_frac * r_frac * (1 - c_frac));             // p110
        accumulate_func(d_floor + 1, r_floor + 1, c_floor + 1, grad_val * d_frac * r_frac * c_frac);                   // p111
    }
};

/**
 * Tricubic backward kernel - distributes gradients to 4x4x4 neighborhood
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct TricubicBackwardKernel : public Backward3DKernel<scalar_t, real_t> {
private:
    // Reuse the same cubic kernel function from the forward implementation
    inline real_t cubic_kernel(real_t s) const {
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
        std::function<void(int64_t, int64_t, int64_t, scalar_t)> accumulate_func,
        scalar_t grad_val,
        real_t d, real_t r, real_t c
    ) const override {
        // Extract integer and fractional parts
        const int64_t d_floor = floor(d);
        const int64_t r_floor = floor(r);
        const int64_t c_floor = floor(c);
        const real_t d_frac = d - d_floor;
        const real_t r_frac = r - r_floor;
        const real_t c_frac = c - c_floor;
        
        // Distribute gradient to 4x4x4 neighborhood using tricubic weights
        // These are exactly the same weights used in forward tricubic interpolation
        for (int k = -1; k <= 2; ++k) {      // Depth offset: covers 4 depths
            const real_t weight_d = cubic_kernel(d_frac - k);

            for (int i = -1; i <= 2; ++i) {  // Row offset: covers 4 rows
                const real_t weight_r = cubic_kernel(r_frac - i);

                for (int j = -1; j <= 2; ++j) {  // Column offset: covers 4 columns
                    const real_t weight_c = cubic_kernel(c_frac - j);
                    const real_t total_weight = weight_d * weight_r * weight_c;
                    
                    // Only distribute if weight is non-zero (cubic has finite support)
                    if (total_weight != 0.0) {
                        accumulate_func(d_floor + k, r_floor + i, c_floor + j, grad_val * total_weight);
                    }
                }
            }
        }
    }
};

/**
 * Factory function to create backward projection kernels for 3D->2D
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
std::unique_ptr<Backward3DKernel<scalar_t, real_t>> get_3d_backward_kernel(const std::string& interpolation) {
    if (interpolation == "linear") {
        return std::make_unique<TrilinearBackwardKernel<scalar_t, real_t>>();
    } else if (interpolation == "cubic") {
        return std::make_unique<TricubicBackwardKernel<scalar_t, real_t>>();
    } else {
        throw std::runtime_error("Unsupported 3D backward interpolation method: " + interpolation);
    }
}

/**
 * Forward projection from 4D Fourier reconstruction to 2D projections
 * 
 * This is the main "gather" operation that samples a 4D Fourier-space reconstruction
 * at rotated coordinate grids to produce 2D projections. This implements the
 * Central Slice Theorem in Fourier space for 3D->2D projections.
 * 
 * Algorithm:
 * 1. For each output pixel (i,j) in the 2D projection
 * 2. Convert to 2D Fourier coordinates (proj_coord_r, proj_coord_c)
 * 3. Extend to 3D central slice: (proj_coord_r, proj_coord_c, 0)
 * 4. Apply 3x3 rotation matrix to get sampling coordinates in 3D reconstruction
 * 5. Apply oversampling scaling if specified
 * 6. Interpolate from 4D reconstruction at these 3D coordinates
 * 7. Apply phase shift if shifts are provided
 * 8. Store result in 2D projection
 */
at::Tensor forward_project_3d_to_2d_cpu(
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
    TORCH_CHECK(reconstruction.dim() == 4, "Reconstruction must be a 4D tensor (B, D, H, W/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 3 && rotations.size(3) == 3, "Rotations must be (B_rot, P, 3, 3)");

    // Extract tensor dimensions
    const auto B = reconstruction.size(0);          // Batch size
    const auto D = reconstruction.size(1);          // Depth of 3D reconstruction
    const auto boxsize = reconstruction.size(2);    // Reconstruction height
    const auto boxsize_half = reconstruction.size(3); // Reconstruction width (FFTW format)
    
    const auto B_rot = rotations.size(0);           // Rotation batch size
    const auto P = rotations.size(1);               // Number of poses per batch
    TORCH_CHECK(B_rot == B || B_rot == 1, "Batch size of rotations must be 1 or same as reconstruction");

    const auto proj_boxsize = output_shape[0];      // Output projection size
    const auto proj_boxsize_half = output_shape[0] / 2 + 1; // Output width (FFTW format)
    
    // Initialize output tensor with zeros
    auto projection = torch::zeros({B, P, proj_boxsize, proj_boxsize_half}, reconstruction.options());

    // PyTorch dispatch macro for type-generic code
    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "forward_project_3d_to_2d_cpu_rotations", ([&] {
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
        AT_DISPATCH_COMPLEX_TYPES(reconstruction.scalar_type(), "forward_project_3d_to_2d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;  // Extract real type from complex
            auto rec_acc = reconstruction.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto proj_acc = projection.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();

            // Set up Fourier space filtering
            const real_t default_radius = proj_boxsize / 2.0;  // Nyquist frequency
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;  // Precompute for efficiency

            auto kernel = get_3d_interpolation_kernel<scalar_t>(interpolation);

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
                            // Convert array indices to 2D Fourier coordinates
                            // FFTW format: DC at (0,0), positive frequencies, then negative
                            real_t proj_coord_c = j;  // Column: always positive (FFTW half-space)
                            real_t proj_coord_r = (i <= proj_boxsize / 2) ? i : i - proj_boxsize; // Row: handle wrap-around

                            // Apply Fourier space filtering (low-pass)
                            if (proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r > radius_cutoff_sq) {
                                continue;  // Skip high frequencies
                            }
                            
                            // Apply oversampling scaling to 2D coordinates
                            // Oversampling > 1 simulates zero-padding in real space
                            real_t sample_c = proj_coord_c * oversampling;
                            real_t sample_r = proj_coord_r * oversampling;
                            
                            // Central slice: extend 2D coordinates to 3D with z=0
                            real_t sample_d = 0.0;  // Central slice through origin
                            
                            // Apply 3x3 rotation matrix to get sampling coordinates in 3D reconstruction
                            // Matrix multiplication: [rot_c; rot_r; rot_d] = R * [sample_c; sample_r; sample_d]
                            real_t rot_c = rot_acc[rot_b_idx][p][0][0] * sample_c + rot_acc[rot_b_idx][p][0][1] * sample_r + rot_acc[rot_b_idx][p][0][2] * sample_d;
                            real_t rot_r = rot_acc[rot_b_idx][p][1][0] * sample_c + rot_acc[rot_b_idx][p][1][1] * sample_r + rot_acc[rot_b_idx][p][1][2] * sample_d;
                            real_t rot_d = rot_acc[rot_b_idx][p][2][0] * sample_c + rot_acc[rot_b_idx][p][2][1] * sample_r + rot_acc[rot_b_idx][p][2][2] * sample_d;

                            // Interpolate from 4D reconstruction at rotated 3D coordinates
                            scalar_t val = kernel->interpolate(rec_acc, b, boxsize, boxsize_half, rot_d, rot_r, rot_c);
                            
                            // Apply phase shift if translations are provided
                            // Shift in real space = phase modulation in Fourier space
                            if (shifts_acc.has_value()) {
                                const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                                // Compute phase: -2π * (k · shift) for 2D shift
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
 * Unified backward projection function for 3D->2D projections with smart gradient computation
 * 
 * This function computes gradients w.r.t. reconstruction, rotations, and shifts
 * based on what actually requires gradients. It automatically detects which
 * parameters need gradients using requires_grad() and only computes those,
 * avoiding unnecessary computation.
 * 
 * Features:
 * 1. Always computes reconstruction gradients (main scatter operation from 2D to 4D)
 * 2. Only computes rotation gradients if rotations.requires_grad() is true
 * 3. Only computes shift gradients if shifts exist and require gradients
 * 4. Uses proper adjoint operations for mathematical consistency
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_project_3d_to_2d_cpu(
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
    TORCH_CHECK(grad_projections.dim() == 4, "Projections must be a 4D tensor (B, P, H, W/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 3 && rotations.size(3) == 3, "Rotations must be (B_rot, P, 3, 3)");

    const auto B = grad_projections.size(0);
    const auto P = grad_projections.size(1);
    const auto proj_boxsize = grad_projections.size(2);
    const auto proj_boxsize_half = grad_projections.size(3);
    
    const auto rec_depth = reconstruction.size(1);
    const auto rec_boxsize = reconstruction.size(2);
    const auto rec_boxsize_half = reconstruction.size(3);
    const auto boxsize = rec_boxsize;  // For consistency with forward pass variable naming
    
    // Always compute reconstruction gradients (this is the main scatter operation)
    auto grad_reconstruction = torch::zeros({B, rec_depth, rec_boxsize, rec_boxsize_half}, grad_projections.options());
    
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

    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "backward_project_3d_to_2d_cpu", ([&] {
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

        AT_DISPATCH_COMPLEX_TYPES(grad_projections.scalar_type(), "backward_project_3d_to_2d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;
            auto grad_proj_acc = grad_projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto grad_rec_acc = grad_reconstruction.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto rec_acc = reconstruction.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();

            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;

            auto kernel = get_3d_interpolation_kernel<scalar_t>(interpolation);
            auto backward_kernel = get_3d_backward_kernel<scalar_t>(interpolation);
            auto kernel_grad = get_3d_interpolation_kernel<scalar_t>(interpolation);
            
            // Parallelize over batch*pose combinations
            const int64_t grain_size = std::max(int64_t(1), (B * P) / (2 * at::get_num_threads()));
            at::parallel_for(0, B * P, grain_size, [&](int64_t start, int64_t end) {
                for (int64_t bp_idx = start; bp_idx < end; ++bp_idx) {
                    const int64_t b = bp_idx / P;
                    const int64_t p = bp_idx % P;
                    
                    // Local accumulator arrays for this projection's gradients
                    rot_real_t local_rot_grad[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};      // 3x3 rotation matrix gradients
                    rot_real_t local_shift_grad[2] = {0, 0};                                    // 2-element shift gradients
                    
                    // Lambda function to safely accumulate gradients with 3D Friedel symmetry
                    auto accumulate_grad = [&](int64_t d, int64_t r, int64_t c, scalar_t grad) {
                        bool needs_conj = false;
                        
                        // Handle 3D Friedel symmetry for negative column indices
                        if (c < 0) { 
                            c = -c;           // Mirror column to positive side
                            r = -r;           // Mirror row as well
                            d = -d;           // Mirror depth as well
                            needs_conj = true; // Need to conjugate the value
                        }
                        
                        // Bounds checking
                        if (c >= rec_boxsize_half) return;  // Beyond stored frequency range
                        if (r > boxsize / 2 || r < -boxsize / 2 + 1) return;  // Beyond valid row range
                        if (d > boxsize / 2 || d < -boxsize / 2 + 1) return;  // Beyond valid depth range

                        // Convert negative indices to positive (FFTW wrapping)
                        int64_t r_eff = r < 0 ? boxsize + r : r;
                        int64_t d_eff = d < 0 ? boxsize + d : d;
                        if (r_eff >= boxsize || d_eff >= boxsize) return;  // Final bounds check

                        // Accumulate gradient (with conjugation if needed for Friedel symmetry)
                        scalar_t final_grad = needs_conj ? std::conj(grad) : grad;
                        atomic_add_complex(&grad_rec_acc[b][d_eff][r_eff][c], final_grad);
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
                            real_t sample_d = 0.0;  // Central slice
                            
                            real_t rot_c = rot_acc[rot_b_idx][p][0][0] * sample_c + rot_acc[rot_b_idx][p][0][1] * sample_r + rot_acc[rot_b_idx][p][0][2] * sample_d;
                            real_t rot_r = rot_acc[rot_b_idx][p][1][0] * sample_c + rot_acc[rot_b_idx][p][1][1] * sample_r + rot_acc[rot_b_idx][p][1][2] * sample_d;
                            real_t rot_d = rot_acc[rot_b_idx][p][2][0] * sample_c + rot_acc[rot_b_idx][p][2][1] * sample_r + rot_acc[rot_b_idx][p][2][2] * sample_d;
                            
                            // Use 3D interpolation kernel for forward evaluation (if needed for shifts/rotations)
                            scalar_t rec_val = kernel->interpolate(rec_acc, b, rec_boxsize, rec_boxsize_half, rot_d, rot_r, rot_c);
                            scalar_t grad_proj = grad_proj_acc[b][p][i][j];

                            // Apply phase shift correction to gradient if shifts are present
                            if (shifts_acc.has_value()) {
                               const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                               real_t phase = 2.0 * M_PI * (proj_coord_r * (*shifts_acc)[shift_b_idx][p][0] / boxsize + 
                                                             proj_coord_c * (*shifts_acc)[shift_b_idx][p][1] / boxsize);
                               scalar_t phase_factor = scalar_t(cos(phase), sin(phase));
                               grad_proj = grad_proj * phase_factor;
                            }

                            // Use abstracted 3D backward kernel for gradient distribution
                            // This ensures the backward pass is the proper adjoint of the forward interpolation
                            backward_kernel->distribute_gradient(accumulate_grad, grad_proj, rot_d, rot_r, rot_c);

                            // Compute gradients w.r.t. shift parameters (only if needed)
                            if (need_shift_grads) {
                                const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                                
                                // Apply phase modulation to reconstruction value for correct gradient
                                scalar_t modulated_rec_val = rec_val;
                                if (shifts_acc.has_value()) {
                                    real_t phase = -2.0 * M_PI * (proj_coord_r * (*shifts_acc)[shift_b_idx][p][0] / boxsize + 
                                                                   proj_coord_c * (*shifts_acc)[shift_b_idx][p][1] / boxsize);
                                    scalar_t phase_factor = scalar_t(cos(phase), sin(phase));
                                    modulated_rec_val = rec_val * phase_factor;
                                }
                                
                                // Compute phase derivatives: ∂φ/∂shift = -2π * coordinate / boxsize
                                scalar_t phase_grad_r = scalar_t(0, -2.0 * M_PI * proj_coord_r / boxsize) * modulated_rec_val;
                                scalar_t phase_grad_c = scalar_t(0, -2.0 * M_PI * proj_coord_c / boxsize) * modulated_rec_val;
                                
                                // Accumulate shift gradients locally (take real part of complex gradient)
                                scalar_t original_grad_proj = grad_proj_acc[b][p][i][j];
                                local_shift_grad[0] += (original_grad_proj * std::conj(phase_grad_r)).real();
                                local_shift_grad[1] += (original_grad_proj * std::conj(phase_grad_c)).real();
                            }

                            // Compute gradients w.r.t. 3x3 rotation matrix elements (only if needed)
                            if (need_rotation_grads) {
                                auto [rec_val_unused, grad_d, grad_r, grad_c] = kernel_grad->interpolate_with_gradients(rec_acc, b, rec_boxsize, rec_boxsize_half, rot_d, rot_r, rot_c);
                                
                                // Chain rule: ∂f/∂R[i][j] = (∂f/∂rot_coord) * (∂rot_coord/∂R[i][j])
                                // 
                                // 3D Rotation transformation: [rot_c; rot_r; rot_d] = R * [sample_c; sample_r; sample_d]
                                // Therefore:
                                // ∂rot_c/∂R[0][0] = sample_c, ∂rot_c/∂R[0][1] = sample_r, ∂rot_c/∂R[0][2] = sample_d
                                // ∂rot_r/∂R[1][0] = sample_c, ∂rot_r/∂R[1][1] = sample_r, ∂rot_r/∂R[1][2] = sample_d
                                // ∂rot_d/∂R[2][0] = sample_c, ∂rot_d/∂R[2][1] = sample_r, ∂rot_d/∂R[2][2] = sample_d
                                //
                                // Accumulate gradients locally (taking real part for real-valued rotation matrices):
                                local_rot_grad[0][0] += (grad_proj * std::conj(grad_c * sample_c)).real();  // ∂f/∂R[0][0]
                                local_rot_grad[0][1] += (grad_proj * std::conj(grad_c * sample_r)).real();  // ∂f/∂R[0][1]
                                local_rot_grad[0][2] += (grad_proj * std::conj(grad_c * sample_d)).real();  // ∂f/∂R[0][2]
                                local_rot_grad[1][0] += (grad_proj * std::conj(grad_r * sample_c)).real();  // ∂f/∂R[1][0]
                                local_rot_grad[1][1] += (grad_proj * std::conj(grad_r * sample_r)).real();  // ∂f/∂R[1][1]
                                local_rot_grad[1][2] += (grad_proj * std::conj(grad_r * sample_d)).real();  // ∂f/∂R[1][2]
                                local_rot_grad[2][0] += (grad_proj * std::conj(grad_d * sample_c)).real();  // ∂f/∂R[2][0]
                                local_rot_grad[2][1] += (grad_proj * std::conj(grad_d * sample_r)).real();  // ∂f/∂R[2][1]
                                local_rot_grad[2][2] += (grad_proj * std::conj(grad_d * sample_d)).real();  // ∂f/∂R[2][2]
                            }
                        }
                    }
                    
                    // Add local gradients to global tensors
                    if (need_rotation_grads && grad_rot_acc.has_value()) {
                        const int64_t rot_b_idx = (rotations.size(0) == 1) ? 0 : b;
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                atomic_add_real(&(*grad_rot_acc)[rot_b_idx][p][i][j], local_rot_grad[i][j]);
                            }
                        }
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