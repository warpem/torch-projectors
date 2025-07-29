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
    int64_t b, int64_t boxsize, int64_t boxsize_half,
    int64_t rec_batch_stride, int64_t rec_row_stride,
    int64_t r, int64_t c
) {
    bool need_conjugate = false;
    
    // Handle negative kx via Friedel symmetry (c < 0)
    if (c < 0) {
        c = -c;
        r = -r;
        need_conjugate = !need_conjugate;
    }
    
    // Clamp coordinates to valid array bounds
    c = min(c, boxsize_half - 1);
    r = min(boxsize / 2, max(r, -boxsize / 2 + 1));
    
    // Convert negative row indices to positive (FFTW wrapping)
    if (r < 0) {
        r = boxsize + r;
    }
    r = min(r, boxsize - 1);
    
    // Calculate linear index
    int64_t idx = b * rec_batch_stride + r * rec_row_stride + c;
    float2 value = rec[idx];
    
    // Return conjugated value if we used Friedel symmetry
    return need_conjugate ? complex_conj(value) : value;
}

// Bilinear interpolation kernel
inline float2 bilinear_interpolate(
    device const float2* rec,
    int64_t b, int64_t boxsize, int64_t boxsize_half,
    int64_t rec_batch_stride, int64_t rec_row_stride,
    float r, float c
) {
    // Extract integer and fractional parts
    int64_t c_floor = floor(c);
    int64_t r_floor = floor(r);
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
    int64_t b, int64_t boxsize, int64_t boxsize_half,
    int64_t rec_batch_stride, int64_t rec_row_stride,
    int64_t r, int64_t c
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
    int64_t b, int64_t boxsize, int64_t boxsize_half,
    int64_t rec_batch_stride, int64_t rec_row_stride,
    float r, float c
) {
    // Extract integer and fractional parts
    int64_t c_floor = floor(c);
    int64_t r_floor = floor(r);
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

// Main forward projection compute kernel
kernel void forward_project_2d_kernel(
    device const float2* reconstruction [[buffer(0)]],
    device const float* rotations [[buffer(1)]],
    device const float* shifts [[buffer(2)]],
    device float2* projections [[buffer(3)]],
    constant int64_t& B [[buffer(4)]],
    constant int64_t& P [[buffer(5)]],
    constant int64_t& boxsize [[buffer(6)]],
    constant int64_t& boxsize_half [[buffer(7)]],
    constant int64_t& proj_boxsize [[buffer(8)]],
    constant int64_t& proj_boxsize_half [[buffer(9)]],
    constant int64_t& B_rot [[buffer(10)]],
    constant bool& has_shifts [[buffer(11)]],
    constant int64_t& B_shift [[buffer(12)]],
    constant int& interpolation_method [[buffer(13)]],  // 0=linear, 1=cubic
    constant float& oversampling [[buffer(14)]],
    constant float& fourier_radius_cutoff [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Thread organization: gid.x = column (j), gid.y = row (i), gid.z = batch*pose
    int64_t j = gid.x;
    int64_t i = gid.y;
    int64_t bp_idx = gid.z;
    
    // Check bounds
    if (j >= proj_boxsize_half || i >= proj_boxsize || bp_idx >= B * P) {
        return;
    }
    
    // Extract batch and pose indices
    int64_t b = bp_idx / P;
    int64_t p = bp_idx % P;
    
    // Calculate strides for memory access
    int64_t rec_batch_stride = boxsize * boxsize_half;
    int64_t rec_row_stride = boxsize_half;
    int64_t rot_batch_stride = P * 2 * 2;
    int64_t rot_pose_stride = 2 * 2;
    int64_t proj_batch_stride = P * proj_boxsize * proj_boxsize_half;
    int64_t proj_pose_stride = proj_boxsize * proj_boxsize_half;
    int64_t proj_row_stride = proj_boxsize_half;
    
    // Convert array indices to Fourier coordinates
    float proj_coord_c = j;
    float proj_coord_r = (i <= proj_boxsize / 2) ? i : i - proj_boxsize;
    
    // Apply Fourier space filtering (low-pass)
    float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
    float radius_cutoff_sq = fourier_radius_cutoff * fourier_radius_cutoff;
    if (radius_sq > radius_cutoff_sq) {
        // Set to zero for filtered frequencies
        int64_t proj_idx = b * proj_batch_stride + p * proj_pose_stride + i * proj_row_stride + j;
        projections[proj_idx] = float2(0.0, 0.0);
        return;
    }
    
    // Apply oversampling scaling to coordinates
    float sample_c = proj_coord_c * oversampling;
    float sample_r = proj_coord_r * oversampling;
    
    // Apply rotation matrix to get sampling coordinates in reconstruction
    int64_t rot_b_idx = (B_rot == 1) ? 0 : b;
    int64_t rot_idx_base = rot_b_idx * rot_batch_stride + p * rot_pose_stride;
    
    float rot_00 = rotations[rot_idx_base + 0];  // R[0][0]
    float rot_01 = rotations[rot_idx_base + 1];  // R[0][1]
    float rot_10 = rotations[rot_idx_base + 2];  // R[1][0]
    float rot_11 = rotations[rot_idx_base + 3];  // R[1][1]
    
    float rot_c = rot_00 * sample_c + rot_01 * sample_r;
    float rot_r = rot_10 * sample_c + rot_11 * sample_r;
    
    // Interpolate from reconstruction at rotated coordinates
    float2 val;
    if (interpolation_method == 0) {  // Linear
        val = bilinear_interpolate(reconstruction, b, boxsize, boxsize_half,
                                 rec_batch_stride, rec_row_stride, rot_r, rot_c);
    } else {  // Cubic
        val = bicubic_interpolate(reconstruction, b, boxsize, boxsize_half,
                                rec_batch_stride, rec_row_stride, rot_r, rot_c);
    }
    
    // Apply phase shift if translations are provided
    if (has_shifts) {
        int64_t shift_b_idx = (B_shift == 1) ? 0 : b;
        int64_t shift_idx_base = shift_b_idx * P * 2 + p * 2;
        
        float shift_r = shifts[shift_idx_base + 0];
        float shift_c = shifts[shift_idx_base + 1];
        
        // Compute phase: -2π * (k · shift)
        float phase = -2.0 * M_PI_F * (proj_coord_r * shift_r / boxsize + proj_coord_c * shift_c / boxsize);
        float2 phase_factor = float2(cos(phase), sin(phase));  // e^(iφ)
        val = complex_mul(val, phase_factor);
    }
    
    // Store result in projection
    int64_t proj_idx = b * proj_batch_stride + p * proj_pose_stride + i * proj_row_stride + j;
    projections[proj_idx] = val;
}