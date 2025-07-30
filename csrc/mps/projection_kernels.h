#pragma once

#ifdef __APPLE__

// Embedded Metal shader source for forward projection kernel
// This is the standard approach for PyTorch extensions to include shaders
constexpr const char* PROJECTION_KERNEL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// Complex number operations using float2 (real, imag)
inline float2 complex_mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 complex_conj(float2 a) {
    return float2(a.x, -a.y);
}

inline float2 complex_add(float2 a, float2 b) {
    return a + b;
}

inline float2 complex_scale(float2 a, float s) {
    return a * s;
}

// Sample from FFTW-formatted Fourier space with automatic Friedel symmetry handling
// Ported from sample_fftw_with_conjugate in CPU version
inline float2 sample_fftw_with_conjugate(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    int32_t r, int32_t c
) {
    bool need_conjugate = false;
    
    // Handle negative kx via Friedel symmetry (c < 0)
    if (c < 0) {
        c = -c;
        r = -r;
        need_conjugate = !need_conjugate;
    }
    
    // Clamp coordinates to valid array bounds
    c = min(c, (int32_t)boxsize_half - 1);
    r = min((int32_t)boxsize / 2, max(r, -(int32_t)boxsize / 2 + 1));
    
    // Convert negative row indices to positive (FFTW wrapping)
    if (r < 0) {
        r = (int32_t)boxsize + r;
    }
    r = min(r, (int32_t)boxsize - 1);
    
    // Calculate linear index
    int32_t idx = b * rec_batch_stride + r * rec_row_stride + c;
    float2 value = rec[idx];
    
    // Return conjugated value if we used Friedel symmetry
    return need_conjugate ? complex_conj(value) : value;
}

// Bilinear interpolation kernel
inline float2 bilinear_interpolate(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Sample 2x2 grid of neighboring pixels
    float2 p00 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor, c_floor);
    float2 p01 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor, c_floor + 1);
    float2 p10 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor + 1, c_floor);
    float2 p11 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor + 1, c_floor + 1);
    
    // Bilinear interpolation
    float2 p0 = complex_add(p00, complex_scale(complex_add(p01, -p00), c_frac));
    float2 p1 = complex_add(p10, complex_scale(complex_add(p11, -p10), c_frac));
    return complex_add(p0, complex_scale(complex_add(p1, -p0), r_frac));
}

// Bicubic interpolation kernel helper functions
inline float bicubic_kernel(float s) {
    const float a = -0.5;  // Catmull-Rom parameter
    s = abs(s);
    
    if (s <= 1.0) {
        return (a + 2.0) * s * s * s - (a + 3.0) * s * s + 1.0;
    } else if (s <= 2.0) {
        return a * s * s * s - 5.0 * a * s * s + 8.0 * a * s - 4.0 * a;
    } else {
        return 0.0;
    }
}

// Safe sampling with edge clamping for bicubic interpolation
inline float2 sample_with_edge_clamping(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    int32_t r, int32_t c
) {
    // Clamp coordinates to valid ranges
    if (abs(c) >= boxsize_half) {
        c = (c < 0) ? -(boxsize_half - 1) : (boxsize_half - 1);
    }
    r = max(-boxsize / 2 + 1, min(r, boxsize / 2));
    
    return sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r, c);
}

// Bicubic interpolation kernel
inline float2 bicubic_interpolate(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    float2 result = float2(0.0, 0.0);
    
    // Sample 4x4 grid around the interpolation point
    for (int i = -1; i <= 2; ++i) {
        float weight_r = bicubic_kernel(r_frac - i);
        
        for (int j = -1; j <= 2; ++j) {
            float2 sample = sample_with_edge_clamping(rec, b, boxsize, boxsize_half, 
                                                    rec_batch_stride, rec_row_stride,
                                                    r_floor + i, c_floor + j);
            float weight_c = bicubic_kernel(c_frac - j);
            result = complex_add(result, complex_scale(sample, weight_r * weight_c));
        }
    }
    
    return result;
}

struct Params {
    int B, P, boxsize, boxsize_half;
    int proj_boxsize, proj_boxsize_half, B_rot, B_shift;
    int has_shifts;
    int interpolation_method;
    float oversampling;
    float fourier_radius_cutoff;
};

// Main forward projection compute kernel - OPTIMIZED VERSION
kernel void forward_project_2d_kernel(
    device const float2* reconstruction [[buffer(0)]],
    device const float*  rotations      [[buffer(1)]],
    device const float*  shifts         [[buffer(2)]],
    device float2*       projections    [[buffer(3)]],
    constant Params&     params         [[buffer(4)]],
    uint2 gpos [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint2 tpg  [[threads_per_threadgroup]]
) {
    // Optimized thread organization: each threadgroup handles one entire projection
    // Grid: (P, B, 1), Threadgroup: (256, 1, 1)
    int32_t p = gpos.x;  // Pose index
    int32_t b = gpos.y;  // Batch index
    
    // Check bounds for batch and pose
    if (p >= params.P || b >= params.B) {
        return;
    }
    
    // Pre-compute shared parameters for this projection (pose p, batch b)
    
    // Rotation matrix (shared for all pixels in this projection)
    int rot_b_idx = (params.B_rot == 1) ? 0 : b;
    int rot_idx_base = ((rot_b_idx * params.P + p) * 2 * 2);  // Index to start of 2x2 matrix
    float rot_00 = rotations[rot_idx_base + 0];     // R[0][0]
    float rot_01 = rotations[rot_idx_base + 1];     // R[0][1] 
    float rot_10 = rotations[rot_idx_base + 2];     // R[1][0]
    float rot_11 = rotations[rot_idx_base + 3];     // R[1][1]
    
    // Shifts (shared for all pixels in this projection)
    float shift_r = 0.0, shift_c = 0.0;
    if (params.has_shifts) {
        int shift_b_idx = (params.B_shift == 1) ? 0 : b;
        int shift_idx_base = ((shift_b_idx * params.P + p) * 2);
        shift_r = shifts[shift_idx_base + 0];
        shift_c = shifts[shift_idx_base + 1];
    }
    
    // Pre-compute constants for this projection
    int32_t proj_batch_stride = params.P * params.proj_boxsize * params.proj_boxsize_half;
    int32_t proj_pose_stride = params.proj_boxsize * params.proj_boxsize_half;
    int32_t proj_row_stride = params.proj_boxsize_half;
    int32_t proj_base_idx = b * proj_batch_stride + p * proj_pose_stride;
    
    int32_t rec_batch_stride = params.boxsize * params.boxsize_half;
    int32_t rec_row_stride = params.boxsize_half;
    
    float fourier_cutoff_sq = params.fourier_radius_cutoff * params.fourier_radius_cutoff;
    
    // Loop over all pixels in this projection, with threads cooperating
    int32_t total_pixels = params.proj_boxsize * params.proj_boxsize_half;
    
    for (int32_t pixel_idx = tid; pixel_idx < total_pixels; pixel_idx += tpg.x) {
        // Convert linear pixel index to (i, j) coordinates
        int32_t i = pixel_idx / params.proj_boxsize_half;  // Row
        int32_t j = pixel_idx % params.proj_boxsize_half;  // Column
        
        // Convert array indices to Fourier coordinates (must match CPU logic exactly)
        float proj_coord_c = float(j);  // Column: always positive (FFTW half-space)
        float proj_coord_r = (i <= params.proj_boxsize / 2) ? float(i) : float(i) - float(params.proj_boxsize);
        
        // Apply Fourier space filtering (low-pass)
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            projections[proj_base_idx + i * proj_row_stride + j] = float2(0.0, 0.0);
            continue;
        }
        
        // Apply oversampling scaling to coordinates
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Apply rotation matrix (using pre-computed matrix elements)
        float rot_c = rot_00 * sample_c + rot_01 * sample_r;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r;
        
        // Interpolate from reconstruction at rotated coordinates
        float2 val;
        if (params.interpolation_method == 0) {  // linear
            val = bilinear_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                      rec_batch_stride, rec_row_stride, rot_r, rot_c);
        } else {  // cubic
            val = bicubic_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                     rec_batch_stride, rec_row_stride, rot_r, rot_c);
        }
        
        // Apply phase shift if translations are provided (using pre-computed shifts)
        if (params.has_shifts) {
            float phase = -2.0 * M_PI_F * (proj_coord_r * shift_r / params.boxsize + 
                                           proj_coord_c * shift_c / params.boxsize);
            float2 phase_factor = float2(cos(phase), sin(phase));
            val = complex_mul(val, phase_factor);
        }
        
        projections[proj_base_idx + pixel_idx] = val;
    }
}
)";

#endif // __APPLE__