/**
 * Unified Interpolation Kernel Framework
 * 
 * This header provides a template-based design for interpolation kernels
 * that works for both 2D and 3D cases. It uses compile-time templates
 * to generate specialized code with zero runtime overhead.
 */

#pragma once

#include "cubic_kernels.h"
#include "fftw_sampling.h"
#include <torch/extension.h>
#include <functional>
#include <memory>
#include <string>
#include <tuple>

namespace torch_projectors {
namespace cpu {
namespace common {

/**
 * Interpolation method enumeration
 */
enum class InterpolationMethod {
    LINEAR,  // Linear interpolation (N-dimensional)
    CUBIC    // Cubic interpolation (N-dimensional)
};

/**
 * Abstract base class for N-dimensional interpolation kernels
 * 
 * Template parameter N specifies the dimensionality:
 * - N=2: 2D interpolation (bilinear/bicubic)
 * - N=3: 3D interpolation (trilinear/tricubic)
 */
template <int N, typename scalar_t, typename real_t = typename scalar_t::value_type>
struct InterpolationKernel {
    virtual ~InterpolationKernel() = default;
    
    /**
     * Perform N-dimensional interpolation at continuous coordinates
     * This is a pure virtual function that must be implemented by derived classes
     */
    virtual scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, N + 1, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        const std::array<real_t, N>& coords
    ) const = 0;
    
    /**
     * Perform interpolation with simultaneous gradient computation
     * Returns tuple of (value, grad_0, grad_1, ..., grad_{N-1})
     */
    virtual std::tuple<scalar_t, std::array<scalar_t, N>> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, N + 1, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        const std::array<real_t, N>& coords
    ) const = 0;
};

/**
 * Linear interpolation kernel (N-dimensional)
 * 
 * Specializations:
 * - N=2: Bilinear interpolation (2x2 grid)
 * - N=3: Trilinear interpolation (2x2x2 grid)
 */
template <int N, typename scalar_t, typename real_t = typename scalar_t::value_type>
struct LinearKernel : public InterpolationKernel<N, scalar_t, real_t> {
    // Implementation depends on N - we'll provide specializations
};

/**
 * 2D Linear (Bilinear) interpolation specialization
 */
template <typename scalar_t, typename real_t>
struct LinearKernel<2, scalar_t, real_t> : public InterpolationKernel<2, scalar_t, real_t> {
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        const std::array<real_t, 2>& coords
    ) const override {
        const real_t r = coords[0];
        const real_t c = coords[1];
        
        // Extract integer and fractional parts of coordinates
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);
        const real_t c_frac = c - c_floor;  // Fractional part [0,1)
        const real_t r_frac = r - r_floor;  // Fractional part [0,1)

        // Sample 2x2 grid of neighboring pixels
        const scalar_t p00 = sample_fftw_2d(rec, b, boxsize, boxsize_half, r_floor, c_floor);
        const scalar_t p01 = sample_fftw_2d(rec, b, boxsize, boxsize_half, r_floor, c_floor + 1);
        const scalar_t p10 = sample_fftw_2d(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor);
        const scalar_t p11 = sample_fftw_2d(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor + 1);

        // Bilinear interpolation
        const scalar_t p0 = p00 + (p01 - p00) * c_frac;  // Bottom edge
        const scalar_t p1 = p10 + (p11 - p10) * c_frac;  // Top edge
        return p0 + (p1 - p0) * r_frac;  // Final vertical interpolation
    }
    
    std::tuple<scalar_t, std::array<scalar_t, 2>> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        const std::array<real_t, 2>& coords
    ) const override {
        const real_t r = coords[0];
        const real_t c = coords[1];
        
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);
        const real_t c_frac = c - c_floor;
        const real_t r_frac = r - r_floor;

        const scalar_t p00 = sample_fftw_2d(rec, b, boxsize, boxsize_half, r_floor, c_floor);
        const scalar_t p01 = sample_fftw_2d(rec, b, boxsize, boxsize_half, r_floor, c_floor + 1);
        const scalar_t p10 = sample_fftw_2d(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor);
        const scalar_t p11 = sample_fftw_2d(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor + 1);

        // Value computation
        const scalar_t p0 = p00 + (p01 - p00) * c_frac;
        const scalar_t p1 = p10 + (p11 - p10) * c_frac;
        const scalar_t val = p0 + (p1 - p0) * r_frac;

        // Analytical gradients
        const scalar_t grad_r = (1 - c_frac) * (p10 - p00) + c_frac * (p11 - p01);
        const scalar_t grad_c = (1 - r_frac) * (p01 - p00) + r_frac * (p11 - p10);

        return std::make_tuple(val, std::array<scalar_t, 2>{grad_r, grad_c});
    }
};

/**
 * 3D Linear (Trilinear) interpolation specialization
 */
template <typename scalar_t, typename real_t>
struct LinearKernel<3, scalar_t, real_t> : public InterpolationKernel<3, scalar_t, real_t> {
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        const std::array<real_t, 3>& coords
    ) const override {
        const real_t d = coords[0];
        const real_t r = coords[1];
        const real_t c = coords[2];
        
        // Extract integer and fractional parts
        const int64_t d_floor = floor(d);
        const int64_t r_floor = floor(r);
        const int64_t c_floor = floor(c);
        const real_t d_frac = d - d_floor;
        const real_t r_frac = r - r_floor;
        const real_t c_frac = c - c_floor;

        // Sample 2x2x2 = 8 neighboring voxels
        const scalar_t p000 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor,     r_floor,     c_floor);
        const scalar_t p001 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor,     r_floor,     c_floor + 1);
        const scalar_t p010 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor,     r_floor + 1, c_floor);
        const scalar_t p011 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor,     r_floor + 1, c_floor + 1);
        const scalar_t p100 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor,     c_floor);
        const scalar_t p101 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor,     c_floor + 1);
        const scalar_t p110 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor + 1, c_floor);
        const scalar_t p111 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor + 1, c_floor + 1);

        // Trilinear interpolation
        const scalar_t p00 = p000 + (p001 - p000) * c_frac;
        const scalar_t p01 = p010 + (p011 - p010) * c_frac;
        const scalar_t p10 = p100 + (p101 - p100) * c_frac;
        const scalar_t p11 = p110 + (p111 - p110) * c_frac;
        
        const scalar_t p0 = p00 + (p01 - p00) * r_frac;
        const scalar_t p1 = p10 + (p11 - p10) * r_frac;
        
        return p0 + (p1 - p0) * d_frac;
    }
    
    std::tuple<scalar_t, std::array<scalar_t, 3>> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        const std::array<real_t, 3>& coords
    ) const override {
        const real_t d = coords[0];
        const real_t r = coords[1];
        const real_t c = coords[2];
        
        const int64_t d_floor = floor(d);
        const int64_t r_floor = floor(r);
        const int64_t c_floor = floor(c);
        const real_t d_frac = d - d_floor;
        const real_t r_frac = r - r_floor;
        const real_t c_frac = c - c_floor;

        // Sample 2x2x2 grid
        const scalar_t p000 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor,     r_floor,     c_floor);
        const scalar_t p001 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor,     r_floor,     c_floor + 1);
        const scalar_t p010 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor,     r_floor + 1, c_floor);
        const scalar_t p011 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor,     r_floor + 1, c_floor + 1);
        const scalar_t p100 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor,     c_floor);
        const scalar_t p101 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor,     c_floor + 1);
        const scalar_t p110 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor + 1, c_floor);
        const scalar_t p111 = sample_fftw_3d(rec, b, boxsize, boxsize_half, d_floor + 1, r_floor + 1, c_floor + 1);

        // Value computation
        const scalar_t p00 = p000 + (p001 - p000) * c_frac;
        const scalar_t p01 = p010 + (p011 - p010) * c_frac;
        const scalar_t p10 = p100 + (p101 - p100) * c_frac;
        const scalar_t p11 = p110 + (p111 - p110) * c_frac;
        
        const scalar_t p0 = p00 + (p01 - p00) * r_frac;
        const scalar_t p1 = p10 + (p11 - p10) * r_frac;
        
        const scalar_t val = p0 + (p1 - p0) * d_frac;

        // Analytical gradients
        const scalar_t grad_d = p1 - p0;
        const scalar_t grad_r = (1 - d_frac) * (p01 - p00) + d_frac * (p11 - p10);
        const scalar_t grad_c = (1 - d_frac) * (1 - r_frac) * (p001 - p000) +
                                (1 - d_frac) * r_frac * (p011 - p010) +
                                d_frac * (1 - r_frac) * (p101 - p100) +
                                d_frac * r_frac * (p111 - p110);

        return std::make_tuple(val, std::array<scalar_t, 3>{grad_d, grad_r, grad_c});
    }
};

/**
 * Cubic interpolation kernel (N-dimensional)
 * Uses separable cubic kernels for efficiency
 */
template <int N, typename scalar_t, typename real_t = typename scalar_t::value_type>
struct CubicKernel : public InterpolationKernel<N, scalar_t, real_t> {
    // Implementation depends on N - we'll provide specializations
};

/**
 * 2D Cubic (Bicubic) interpolation specialization
 */
template <typename scalar_t, typename real_t>
struct CubicKernel<2, scalar_t, real_t> : public InterpolationKernel<2, scalar_t, real_t> {
private:
    inline scalar_t sample_with_edge_clamping(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        int64_t r, int64_t c
    ) const {
        if (std::abs(c) >= boxsize_half) {
            c = (c < 0) ? -(boxsize_half - 1) : (boxsize_half - 1);
        }
        r = std::max(-boxsize / 2 + 1, std::min(r, boxsize / 2));
        return sample_fftw_2d(rec, b, boxsize, boxsize_half, r, c);
    }

public:
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        const std::array<real_t, 2>& coords
    ) const override {
        const real_t r = coords[0];
        const real_t c = coords[1];
        
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);
        const real_t c_frac = c - c_floor;
        const real_t r_frac = r - r_floor;
        
        scalar_t result = scalar_t(0);
        
        // Sample 4x4 grid using separable cubic kernels
        for (int i = -1; i <= 2; ++i) {
            const real_t weight_r = cubic_kernel(r_frac - i);
            for (int j = -1; j <= 2; ++j) {
                const scalar_t sample = sample_with_edge_clamping(rec, b, boxsize, boxsize_half, 
                                                     r_floor + i, c_floor + j);
                const real_t weight_c = cubic_kernel(c_frac - j);
                result += sample * weight_r * weight_c;
            }
        }
        
        return result;
    }
    
    std::tuple<scalar_t, std::array<scalar_t, 2>> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        const std::array<real_t, 2>& coords
    ) const override {
        const real_t r = coords[0];
        const real_t c = coords[1];
        
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);
        const real_t c_frac = c - c_floor;
        const real_t r_frac = r - r_floor;
        
        scalar_t val = scalar_t(0);
        scalar_t grad_r = scalar_t(0);
        scalar_t grad_c = scalar_t(0);
        
        // Sample 4x4 grid and compute value + gradients
        for (int i = -1; i <= 2; ++i) {
            const real_t weight_r = cubic_kernel(r_frac - i);
            const real_t dweight_r = cubic_kernel_derivative(r_frac - i);
            
            for (int j = -1; j <= 2; ++j) {
                const scalar_t sample = sample_with_edge_clamping(rec, b, boxsize, boxsize_half, 
                                                     r_floor + i, c_floor + j);
                const real_t weight_c = cubic_kernel(c_frac - j);
                const real_t dweight_c = cubic_kernel_derivative(c_frac - j);
                
                val += sample * weight_r * weight_c;
                grad_r += sample * dweight_r * weight_c;
                grad_c += sample * weight_r * dweight_c;
            }
        }
        
        return std::make_tuple(val, std::array<scalar_t, 2>{grad_r, grad_c});
    }
};

/**
 * 3D Cubic (Tricubic) interpolation specialization
 */
template <typename scalar_t, typename real_t>
struct CubicKernel<3, scalar_t, real_t> : public InterpolationKernel<3, scalar_t, real_t> {
private:
    inline scalar_t sample_with_edge_clamping_3d(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        int64_t d, int64_t r, int64_t c
    ) const {
        if (std::abs(c) >= boxsize_half) {
            c = (c < 0) ? -(boxsize_half - 1) : (boxsize_half - 1);
        }
        r = std::max(-boxsize / 2 + 1, std::min(r, boxsize / 2));
        d = std::max(-boxsize / 2 + 1, std::min(d, boxsize / 2));
        return sample_fftw_3d(rec, b, boxsize, boxsize_half, d, r, c);
    }

public:
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        const std::array<real_t, 3>& coords
    ) const override {
        const real_t d = coords[0];
        const real_t r = coords[1];
        const real_t c = coords[2];
        
        const int64_t d_floor = floor(d);
        const int64_t r_floor = floor(r);
        const int64_t c_floor = floor(c);
        const real_t d_frac = d - d_floor;
        const real_t r_frac = r - r_floor;
        const real_t c_frac = c - c_floor;
        
        scalar_t result = scalar_t(0);
        
        // Sample 4x4x4 = 64 grid using separable cubic kernels
        for (int k = -1; k <= 2; ++k) {
            const real_t weight_d = cubic_kernel(d_frac - k);
            for (int i = -1; i <= 2; ++i) {
                const real_t weight_r = cubic_kernel(r_frac - i);
                for (int j = -1; j <= 2; ++j) {
                    const scalar_t sample = sample_with_edge_clamping_3d(rec, b, boxsize, boxsize_half,
                                                                        d_floor + k, r_floor + i, c_floor + j);
                    const real_t weight_c = cubic_kernel(c_frac - j);
                    result += sample * weight_d * weight_r * weight_c;
                }
            }
        }
        
        return result;
    }
    
    std::tuple<scalar_t, std::array<scalar_t, 3>> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        const std::array<real_t, 3>& coords
    ) const override {
        const real_t d = coords[0];
        const real_t r = coords[1];
        const real_t c = coords[2];
        
        const int64_t d_floor = floor(d);
        const int64_t r_floor = floor(r);
        const int64_t c_floor = floor(c);
        const real_t d_frac = d - d_floor;
        const real_t r_frac = r - r_floor;
        const real_t c_frac = c - c_floor;
        
        scalar_t val = scalar_t(0);
        scalar_t grad_d = scalar_t(0);
        scalar_t grad_r = scalar_t(0);
        scalar_t grad_c = scalar_t(0);
        
        // Sample 4x4x4 grid and compute value + gradients
        for (int k = -1; k <= 2; ++k) {
            const real_t weight_d = cubic_kernel(d_frac - k);
            const real_t dweight_d = cubic_kernel_derivative(d_frac - k);
            
            for (int i = -1; i <= 2; ++i) {
                const real_t weight_r = cubic_kernel(r_frac - i);
                const real_t dweight_r = cubic_kernel_derivative(r_frac - i);
                
                for (int j = -1; j <= 2; ++j) {
                    const scalar_t sample = sample_with_edge_clamping_3d(rec, b, boxsize, boxsize_half,
                                                                        d_floor + k, r_floor + i, c_floor + j);
                    const real_t weight_c = cubic_kernel(c_frac - j);
                    const real_t dweight_c = cubic_kernel_derivative(c_frac - j);
                    
                    val += sample * weight_d * weight_r * weight_c;
                    grad_d += sample * dweight_d * weight_r * weight_c;
                    grad_r += sample * weight_d * dweight_r * weight_c;
                    grad_c += sample * weight_d * weight_r * dweight_c;
                }
            }
        }
        
        return std::make_tuple(val, std::array<scalar_t, 3>{grad_d, grad_r, grad_c});
    }
};

/**
 * Factory function to create N-dimensional interpolation kernels
 */
template <int N, typename scalar_t, typename real_t = typename scalar_t::value_type>
std::unique_ptr<InterpolationKernel<N, scalar_t, real_t>> get_interpolation_kernel(const std::string& interpolation) {
    if (interpolation == "linear") {
        return std::make_unique<LinearKernel<N, scalar_t, real_t>>();
    } else if (interpolation == "cubic") {
        return std::make_unique<CubicKernel<N, scalar_t, real_t>>();
    } else {
        throw std::runtime_error("Unsupported interpolation method: " + interpolation);
    }
}

} // namespace common
} // namespace cpu
} // namespace torch_projectors