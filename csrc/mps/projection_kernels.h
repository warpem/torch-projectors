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

// Bilinear interpolation with gradients (for rotation gradient computation)
inline void bilinear_interpolate_with_gradients(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float r, float c,
    thread float2* val, thread float2* grad_r, thread float2* grad_c
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
    
    // Value computation (same as interpolate method)
    float2 p0 = complex_add(p00, complex_scale(complex_add(p01, -p00), c_frac));
    float2 p1 = complex_add(p10, complex_scale(complex_add(p11, -p10), c_frac));
    *val = complex_add(p0, complex_scale(complex_add(p1, -p0), r_frac));
    
    // Analytical spatial gradients derived from bilinear formula
    *grad_r = complex_add(complex_scale(complex_add(p10, -p00), 1.0 - c_frac), 
                         complex_scale(complex_add(p11, -p01), c_frac));
    *grad_c = complex_add(complex_scale(complex_add(p01, -p00), 1.0 - r_frac), 
                         complex_scale(complex_add(p11, -p10), r_frac));
}

// Bicubic kernel derivative for gradient computation
inline float bicubic_kernel_derivative(float s) {
    const float a = -0.5;
    float sign = (s < 0) ? -1.0 : 1.0;
    s = abs(s);
    
    if (s <= 1.0) {
        return sign * (3.0 * (a + 2.0) * s * s - 2.0 * (a + 3.0) * s);
    } else if (s <= 2.0) {
        return sign * (3.0 * a * s * s - 10.0 * a * s + 8.0 * a);
    } else {
        return 0.0;
    }
}

// Bicubic interpolation with gradients (for rotation gradient computation)
inline void bicubic_interpolate_with_gradients(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float r, float c,
    thread float2* val, thread float2* grad_r, thread float2* grad_c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    *val = float2(0.0, 0.0);
    *grad_r = float2(0.0, 0.0);
    *grad_c = float2(0.0, 0.0);
    
    // Sample 4x4 grid and compute value + gradients simultaneously
    for (int i = -1; i <= 2; ++i) {
        float weight_r = bicubic_kernel(r_frac - i);
        float dweight_r = bicubic_kernel_derivative(r_frac - i);
        
        for (int j = -1; j <= 2; ++j) {
            float2 sample = sample_with_edge_clamping(rec, b, boxsize, boxsize_half, 
                                                    rec_batch_stride, rec_row_stride,
                                                    r_floor + i, c_floor + j);
            
            float weight_c = bicubic_kernel(c_frac - j);
            float dweight_c = bicubic_kernel_derivative(c_frac - j);
            
            // Accumulate value and gradients
            *val = complex_add(*val, complex_scale(sample, weight_r * weight_c));
            *grad_r = complex_add(*grad_r, complex_scale(sample, dweight_r * weight_c));
            *grad_c = complex_add(*grad_c, complex_scale(sample, weight_r * dweight_c));
        }
    }
}

// Helper to convert float to uint for atomic operations
inline uint float_as_uint(float f) {
    return as_type<uint>(f);
}

// Helper to convert uint to float for atomic operations  
inline float uint_as_float(uint u) {
    return as_type<float>(u);
}

// Atomic add for complex numbers using compare-and-swap workaround
inline void atomic_add_complex(device float2* target, float2 value) {
    // Metal doesn't support float atomics and can't take address of vector elements
    // Treat the float2 as an array of 2 floats and access them separately
    device float* float_ptr = reinterpret_cast<device float*>(target);
    device _atomic<uint>* real_atomic = reinterpret_cast<device _atomic<uint>*>(float_ptr + 0);
    device _atomic<uint>* imag_atomic = reinterpret_cast<device _atomic<uint>*>(float_ptr + 1);
    
    // Atomic add for real part using CAS loop
    uint expected_real = atomic_load_explicit(real_atomic, memory_order_relaxed);
    uint desired_real;
    do {
        float current_real = uint_as_float(expected_real);
        float new_real = current_real + value.x;
        desired_real = float_as_uint(new_real);
    } while (!atomic_compare_exchange_weak_explicit(real_atomic, &expected_real, desired_real,
                                                   memory_order_relaxed, memory_order_relaxed));
    
    // Atomic add for imaginary part using CAS loop
    uint expected_imag = atomic_load_explicit(imag_atomic, memory_order_relaxed);
    uint desired_imag;
    do {
        float current_imag = uint_as_float(expected_imag);
        float new_imag = current_imag + value.y;
        desired_imag = float_as_uint(new_imag);
    } while (!atomic_compare_exchange_weak_explicit(imag_atomic, &expected_imag, desired_imag,
                                                   memory_order_relaxed, memory_order_relaxed));
}

// Atomic add for real numbers using compare-and-swap workaround
inline void atomic_add_real(device float* target, float value) {
    // Metal doesn't support float atomics, use CAS with integer operations
    device _atomic<uint>* atomic_target = reinterpret_cast<device _atomic<uint>*>(target);
    
    uint expected = atomic_load_explicit(atomic_target, memory_order_relaxed);
    uint desired;
    do {
        float current = uint_as_float(expected);
        float new_value = current + value;
        desired = float_as_uint(new_value);
    } while (!atomic_compare_exchange_weak_explicit(atomic_target, &expected, desired,
                                                   memory_order_relaxed, memory_order_relaxed));
}

// Helper function to safely accumulate gradients with Friedel symmetry
inline void accumulate_gradient_with_symmetry(
    device float2* grad_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    int32_t row, int32_t col, float2 grad
) {
    bool needs_conj = false;
    
    // Handle Friedel symmetry for negative column indices
    if (col < 0) { 
        col = -col;
        row = -row;
        needs_conj = true;
    }
    
    // Bounds checking
    if (col >= boxsize_half) return;
    if (row > boxsize / 2 || row < -boxsize / 2 + 1) return;
    
    // Convert negative row indices to positive (FFTW wrapping)
    int32_t r_eff = row < 0 ? boxsize + row : row;
    if (r_eff >= boxsize) return;
    
    // Calculate linear index and atomically accumulate
    int32_t idx = b * rec_batch_stride + r_eff * rec_row_stride + col;
    float2 final_grad = needs_conj ? complex_conj(grad) : grad;
    atomic_add_complex(&grad_rec[idx], final_grad);
}

// Distribute bilinear gradient to 2x2 neighborhood
inline void distribute_bilinear_gradient(
    device float2* grad_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float2 grad_val, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute gradient to 2x2 neighborhood with bilinear weights
    accumulate_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                    r_floor,     c_floor,     complex_scale(grad_val, (1.0 - r_frac) * (1.0 - c_frac)));
    accumulate_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                    r_floor,     c_floor + 1, complex_scale(grad_val, (1.0 - r_frac) * c_frac));
    accumulate_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                    r_floor + 1, c_floor,     complex_scale(grad_val, r_frac * (1.0 - c_frac)));
    accumulate_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                    r_floor + 1, c_floor + 1, complex_scale(grad_val, r_frac * c_frac));
}

// Distribute bicubic gradient to 4x4 neighborhood
inline void distribute_bicubic_gradient(
    device float2* grad_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float2 grad_val, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute gradient to 4x4 neighborhood using bicubic weights
    for (int i = -1; i <= 2; ++i) {
        float weight_r = bicubic_kernel(r_frac - i);
        
        for (int j = -1; j <= 2; ++j) {
            float weight_c = bicubic_kernel(c_frac - j);
            float total_weight = weight_r * weight_c;
            
            // Only distribute if weight is non-zero
            if (total_weight != 0.0) {
                accumulate_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                                 r_floor + i, c_floor + j, complex_scale(grad_val, total_weight));
            }
        }
    }
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

// Unified backward projection kernel (matches CPU structure)
kernel void backward_project_2d_kernel(
    device const float2* grad_projections [[buffer(0)]],
    device const float2* reconstruction   [[buffer(1)]],
    device const float*  rotations        [[buffer(2)]],
    device const float*  shifts           [[buffer(3)]],
    device float2*       grad_reconstruction [[buffer(4)]],
    device float*        grad_rotations   [[buffer(5)]],
    device float*        grad_shifts      [[buffer(6)]],
    constant Params&     params           [[buffer(7)]],
    uint2 gpos [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint2 tpg  [[threads_per_threadgroup]]
) {
    // Thread organization: each threadgroup handles one entire projection
    // Grid: (P, B, 1), Threadgroup: (256, 1, 1)
    int32_t p = gpos.x;  // Pose index
    int32_t b = gpos.y;  // Batch index
    
    if (p >= params.P || b >= params.B) {
        return;
    }
    
    // Local accumulators for gradients that need reduction
    threadgroup float local_rot_grad[256][4];    // [thread][matrix_element] 
    threadgroup float local_shift_grad[256][2];  // [thread][shift_component]
    
    // Initialize local accumulators
    local_rot_grad[tid][0] = 0.0;   // R[0][0]
    local_rot_grad[tid][1] = 0.0;   // R[0][1] 
    local_rot_grad[tid][2] = 0.0;   // R[1][0]
    local_rot_grad[tid][3] = 0.0;   // R[1][1]
    local_shift_grad[tid][0] = 0.0; // shift_r
    local_shift_grad[tid][1] = 0.0; // shift_c
    
    // Pre-compute shared parameters for this projection
    int rot_b_idx = (params.B_rot == 1) ? 0 : b;
    int rot_idx_base = ((rot_b_idx * params.P + p) * 2 * 2);
    float rot_00 = rotations[rot_idx_base + 0];
    float rot_01 = rotations[rot_idx_base + 1];
    float rot_10 = rotations[rot_idx_base + 2];
    float rot_11 = rotations[rot_idx_base + 3];
    
    float shift_r = 0.0, shift_c = 0.0;
    if (params.has_shifts) {
        int shift_b_idx = (params.B_shift == 1) ? 0 : b;
        int shift_idx_base = ((shift_b_idx * params.P + p) * 2);
        shift_r = shifts[shift_idx_base + 0];
        shift_c = shifts[shift_idx_base + 1];
    }
    
    // Pre-compute strides
    int32_t proj_batch_stride = params.P * params.proj_boxsize * params.proj_boxsize_half;
    int32_t proj_pose_stride = params.proj_boxsize * params.proj_boxsize_half;
    int32_t proj_base_idx = b * proj_batch_stride + p * proj_pose_stride;
    
    int32_t rec_batch_stride = params.boxsize * params.boxsize_half;
    int32_t rec_row_stride = params.boxsize_half;
    
    float fourier_cutoff_sq = params.fourier_radius_cutoff * params.fourier_radius_cutoff;
    
    // Flags for what gradients to compute (passed via params)
    bool need_rotation_grads = (params.interpolation_method & 0x10) != 0;  // Use bit flag
    bool need_shift_grads = (params.interpolation_method & 0x20) != 0;     // Use bit flag
    int interpolation_method = params.interpolation_method & 0x0F;         // Lower 4 bits
    
    // Loop over all pixels in this projection
    int32_t total_pixels = params.proj_boxsize * params.proj_boxsize_half;
    
    for (int32_t pixel_idx = tid; pixel_idx < total_pixels; pixel_idx += tpg.x) {
        int32_t i = pixel_idx / params.proj_boxsize_half;
        int32_t j = pixel_idx % params.proj_boxsize_half;
        
        // Convert array indices to Fourier coordinates
        float proj_coord_c = float(j);
        float proj_coord_r = (i <= params.proj_boxsize / 2) ? float(i) : float(i) - float(params.proj_boxsize);
        
        // Apply Fourier space filtering
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            continue;
        }
        
        // Apply oversampling scaling
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Apply rotation matrix
        float rot_c = rot_00 * sample_c + rot_01 * sample_r;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r;
        
        // Get gradient from projection
        float2 grad_proj = grad_projections[proj_base_idx + pixel_idx];
        
        // Apply phase shift correction to gradient if needed (same for both rec and rot gradients)
        float2 grad_proj_for_rec = grad_proj;
        if (params.has_shifts) {
            float phase = 2.0 * M_PI_F * (proj_coord_r * shift_r / params.boxsize + 
                                         proj_coord_c * shift_c / params.boxsize);
            float2 phase_factor = float2(cos(phase), sin(phase));
            grad_proj_for_rec = complex_mul(grad_proj_for_rec, phase_factor);
        }
        
        // 1. ALWAYS compute reconstruction gradients (main scatter operation)
        // Use the phase-corrected gradient
        if (interpolation_method == 0) {  // linear
            distribute_bilinear_gradient(grad_reconstruction, b, params.boxsize, params.boxsize_half,
                                       rec_batch_stride, rec_row_stride, grad_proj_for_rec, rot_r, rot_c);
        } else {  // cubic
            distribute_bicubic_gradient(grad_reconstruction, b, params.boxsize, params.boxsize_half,
                                      rec_batch_stride, rec_row_stride, grad_proj_for_rec, rot_r, rot_c);
        }
        
        // 2. Compute rotation gradients (only if needed)
        if (need_rotation_grads) {
            // Get spatial gradients (rec_val not needed for rotation gradients)
            float2 _unused, grad_r, grad_c;
            if (interpolation_method == 0) {  // linear
                bilinear_interpolate_with_gradients(reconstruction, b, params.boxsize, params.boxsize_half,
                                                  rec_batch_stride, rec_row_stride, rot_r, rot_c,
                                                  &_unused, &grad_r, &grad_c);
            } else {  // cubic
                bicubic_interpolate_with_gradients(reconstruction, b, params.boxsize, params.boxsize_half,
                                                 rec_batch_stride, rec_row_stride, rot_r, rot_c,
                                                 &_unused, &grad_r, &grad_c);
            }
            
            // Use the SAME (phase-correct) gradient for rotation as for reconstruction
            const float2 grad_for_rot = grad_proj_for_rec;
            
            // Chain rule: ∂f/∂R[i][j] = (∂f/∂rot_coord) * (∂rot_coord/∂R[i][j])
            // IMPORTANT: Must match CPU version exactly - conjugate AFTER multiplying by sample
            
            local_rot_grad[tid][0] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_c, sample_c)))).x;  // ∂f/∂R[0][0]
            local_rot_grad[tid][1] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_c, sample_r)))).x;  // ∂f/∂R[0][1]
            local_rot_grad[tid][2] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_r, sample_c)))).x;  // ∂f/∂R[1][0]
            local_rot_grad[tid][3] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_r, sample_r)))).x;  // ∂f/∂R[1][1]
        }
        
        // 3. Compute shift gradients (only if needed)
        if (need_shift_grads) {
            // Get reconstruction value
            float2 rec_val;
            if (interpolation_method == 0) {  // linear
                rec_val = bilinear_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                             rec_batch_stride, rec_row_stride, rot_r, rot_c);
            } else {  // cubic
                rec_val = bicubic_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                            rec_batch_stride, rec_row_stride, rot_r, rot_c);
            }
            
            // Apply phase modulation to reconstruction value
            float phase = -2.0 * M_PI_F * (proj_coord_r * shift_r / params.boxsize + 
                                           proj_coord_c * shift_c / params.boxsize);
            float2 phase_factor = float2(cos(phase), sin(phase));
            float2 modulated_rec_val = complex_mul(rec_val, phase_factor);
            
            // Compute phase derivatives: ∂φ/∂shift = -2π * coordinate / boxsize
            float2 phase_grad_r = complex_mul(float2(0.0, -2.0 * M_PI_F * proj_coord_r / params.boxsize), modulated_rec_val);
            float2 phase_grad_c = complex_mul(float2(0.0, -2.0 * M_PI_F * proj_coord_c / params.boxsize), modulated_rec_val);
            
            // Accumulate shift gradients (taking real part)
            local_shift_grad[tid][0] += (complex_mul(grad_proj, complex_conj(phase_grad_r))).x;
            local_shift_grad[tid][1] += (complex_mul(grad_proj, complex_conj(phase_grad_c))).x;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce and write gradients (only thread 0 writes to avoid races)
    if (tid == 0) {
        // Reduce rotation gradients across threads
        if (need_rotation_grads) {
            float total_rot_grad[4] = {0.0, 0.0, 0.0, 0.0};
            for (uint t = 0; t < tpg.x; ++t) {
                total_rot_grad[0] += local_rot_grad[t][0];
                total_rot_grad[1] += local_rot_grad[t][1];
                total_rot_grad[2] += local_rot_grad[t][2];
                total_rot_grad[3] += local_rot_grad[t][3];
            }
            
            // Atomically add to global rotation gradients
            int global_rot_idx_base = ((rot_b_idx * params.P + p) * 2 * 2);
            atomic_add_real(&grad_rotations[global_rot_idx_base + 0], total_rot_grad[0]);
            atomic_add_real(&grad_rotations[global_rot_idx_base + 1], total_rot_grad[1]);
            atomic_add_real(&grad_rotations[global_rot_idx_base + 2], total_rot_grad[2]);
            atomic_add_real(&grad_rotations[global_rot_idx_base + 3], total_rot_grad[3]);
        }
        
        // Reduce shift gradients across threads
        if (need_shift_grads) {
            float total_shift_grad[2] = {0.0, 0.0};
            for (uint t = 0; t < tpg.x; ++t) {
                total_shift_grad[0] += local_shift_grad[t][0];
                total_shift_grad[1] += local_shift_grad[t][1];
            }
            
            // Atomically add to global shift gradients
            int shift_b_idx = (params.B_shift == 1) ? 0 : b;
            int global_shift_idx_base = ((shift_b_idx * params.P + p) * 2);
            atomic_add_real(&grad_shifts[global_shift_idx_base + 0], total_shift_grad[0]);
            atomic_add_real(&grad_shifts[global_shift_idx_base + 1], total_shift_grad[1]);
        }
    }
}
)";

#endif // __APPLE__