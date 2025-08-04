/**
 * FFTW Sampling with Friedel Symmetry Handling
 * 
 * This header provides template-based sampling functions for FFTW-formatted
 * Fourier space data with automatic Friedel symmetry handling. Supports both
 * 2D and 3D sampling through template specialization.
 */

#pragma once

#include <torch/extension.h>
#include <algorithm>
#include <complex>

namespace torch_projectors {
namespace cpu {
namespace common {

/**
 * Template-based FFTW sampling with Friedel symmetry
 * 
 * This template handles both 2D and 3D cases through specialization.
 * FFTW real-to-complex format stores only positive frequencies in the last dimension.
 * For negative frequencies, we use Friedel symmetry: F(-k) = conj(F(k))
 */
template <int N, typename scalar_t, typename real_t = typename scalar_t::value_type>
struct FFTWSampler;

/**
 * 2D FFTW sampling specialization
 * 
 * Sample from FFTW-formatted 2D Fourier space with automatic Friedel symmetry handling
 * 
 * @param rec: 3D complex tensor [batch, height, width/2+1] in FFTW format
 * @param b: batch index
 * @param boxsize: full size of the reconstruction (width before RFFT)
 * @param boxsize_half: width/2+1 (actual stored width)
 * @param r: row coordinate (can be negative, handled via wrapping)
 * @param c: column coordinate (can be negative, handled via Friedel symmetry)
 * @return: Complex value at (r,c) with proper symmetry handling
 */
template <typename scalar_t, typename real_t>
struct FFTWSampler<2, scalar_t, real_t> {
    static inline scalar_t sample(
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
};

/**
 * 3D FFTW sampling specialization
 * 
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
template <typename scalar_t, typename real_t>
struct FFTWSampler<3, scalar_t, real_t> {
    static inline scalar_t sample(
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
};

/**
 * Convenience wrapper functions for template instantiation
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline scalar_t sample_fftw_2d(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
    const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
    int64_t r, int64_t c
) {
    return FFTWSampler<2, scalar_t, real_t>::sample(rec, b, boxsize, boxsize_half, r, c);
}

template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline scalar_t sample_fftw_3d(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
    const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
    int64_t d, int64_t r, int64_t c
) {
    return FFTWSampler<3, scalar_t, real_t>::sample(rec, b, boxsize, boxsize_half, d, r, c);
}

} // namespace common
} // namespace cpu
} // namespace torch_projectors