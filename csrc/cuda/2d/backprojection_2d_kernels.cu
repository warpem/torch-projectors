#include "backprojection_2d_kernels.h"

#ifdef USE_CUDA

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../../cpu/2d/backprojection_2d_kernels.h"

// CUDA utility functions and kernels

struct CudaParams {
    int B, P, boxsize, boxsize_half;
    int proj_boxsize, proj_boxsize_half, B_rot, B_shift;
    int has_shifts;
    int interpolation_method; // 0=linear, 1=cubic, upper bits for gradient flags
    float oversampling;
    float fourier_radius_cutoff;
};

// Complex number operations using cuFloatComplex (float32 only for CUDA kernels)
__device__ __forceinline__ cuFloatComplex complex_mul(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(
        cuCrealf(a) * cuCrealf(b) - cuCimagf(a) * cuCimagf(b),
        cuCrealf(a) * cuCimagf(b) + cuCimagf(a) * cuCrealf(b)
    );
}

__device__ __forceinline__ cuFloatComplex complex_conj(cuFloatComplex a) {
    return make_cuFloatComplex(cuCrealf(a), -cuCimagf(a));
}

__device__ __forceinline__ cuFloatComplex complex_add(cuFloatComplex a, cuFloatComplex b) {
    return cuCaddf(a, b);
}

__device__ __forceinline__ cuFloatComplex complex_scale(cuFloatComplex a, float s) {
    return make_cuFloatComplex(cuCrealf(a) * s, cuCimagf(a) * s);
}

// Atomic operations for safe accumulation
__device__ __forceinline__ void atomic_add_complex(cuFloatComplex* target, cuFloatComplex value) {
    float* real_ptr = reinterpret_cast<float*>(target);
    float* imag_ptr = real_ptr + 1;
    
    atomicAdd(real_ptr, cuCrealf(value));
    atomicAdd(imag_ptr, cuCimagf(value)); 
}

__device__ __forceinline__ void atomic_add_real(float* target, float value) {
    atomicAdd(target, value);
}

// Helper function to safely accumulate complex data with Friedel symmetry
__device__ __forceinline__ void accumulate_data_with_symmetry(
    cuFloatComplex* data_rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    int row, int col, cuFloatComplex data
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
    int r_eff = row < 0 ? boxsize + row : row;
    if (r_eff >= boxsize) return;
    
    // Calculate linear index and atomically accumulate
    int idx = b * rec_batch_stride + r_eff * rec_row_stride + col;
    cuFloatComplex final_data = needs_conj ? complex_conj(data) : data;
    atomic_add_complex(&data_rec[idx], final_data);

    // On the x=0 line, also insert Friedel-symmetric conjugate counterpart
    if (col == 0) {
        int r_eff2 = (-row) < 0 ? boxsize + (-row) : (-row);
        if (r_eff2 >= boxsize || r_eff2 == r_eff) return;

        atomic_add_complex(&data_rec[b * rec_batch_stride + r_eff2 * rec_row_stride + col], complex_conj(final_data));
    }
}

// Helper function to safely accumulate real weights with Friedel symmetry
__device__ __forceinline__ void accumulate_weights_with_symmetry(
    float* weight_rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    int row, int col, float weight
) {
    // Handle Friedel symmetry for negative column indices
    if (col < 0) { 
        col = -col;
        row = -row;
    }
    
    // Bounds checking
    if (col >= boxsize_half) return;
    if (row > boxsize / 2 || row < -boxsize / 2 + 1) return;
    
    // Convert negative row indices to positive (FFTW wrapping)
    int r_eff = row < 0 ? boxsize + row : row;
    if (r_eff >= boxsize) return;
    
    // Calculate linear index and atomically accumulate (weight is always real and positive)
    int idx = b * rec_batch_stride + r_eff * rec_row_stride + col;
    atomic_add_real(&weight_rec[idx], weight);

    // On the x=0 line, also insert Friedel-symmetric counterpart
    if (col == 0) {
        int r_eff2 = (-row) < 0 ? boxsize + (-row) : (-row);
        if (r_eff2 >= boxsize || r_eff2 == r_eff) return;

        atomic_add_real(&weight_rec[b * rec_batch_stride + r_eff2 * rec_row_stride + col], weight);
    }
}

// Distribute bilinear complex data to 2x2 neighborhood
__device__ __forceinline__ void distribute_bilinear_data(
    cuFloatComplex* data_rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    cuFloatComplex data_val, float r, float c
) {
    // Extract integer and fractional parts
    int c_floor = floorf(c);
    int r_floor = floorf(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute data to 2x2 neighborhood with bilinear weights
    accumulate_data_with_symmetry(data_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                r_floor,     c_floor,     complex_scale(data_val, (1.0f - r_frac) * (1.0f - c_frac)));
    accumulate_data_with_symmetry(data_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                r_floor,     c_floor + 1, complex_scale(data_val, (1.0f - r_frac) * c_frac));
    accumulate_data_with_symmetry(data_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                r_floor + 1, c_floor,     complex_scale(data_val, r_frac * (1.0f - c_frac)));
    accumulate_data_with_symmetry(data_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                r_floor + 1, c_floor + 1, complex_scale(data_val, r_frac * c_frac));
}

// Bicubic kernel helper functions
__device__ __forceinline__ float bicubic_kernel(float s) {
    const float a = -0.5f;  // Catmull-Rom parameter
    s = fabsf(s);
    
    if (s <= 1.0f) {
        return (a + 2.0f) * s * s * s - (a + 3.0f) * s * s + 1.0f;
    } else if (s <= 2.0f) {
        return a * s * s * s - 5.0f * a * s * s + 8.0f * a * s - 4.0f * a;
    } else {
        return 0.0f;
    }
}

// Distribute bicubic complex data to 4x4 neighborhood
__device__ __forceinline__ void distribute_bicubic_data(
    cuFloatComplex* data_rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    cuFloatComplex data_val, float r, float c
) {
    // Extract integer and fractional parts
    int c_floor = floorf(c);
    int r_floor = floorf(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute data to 4x4 neighborhood using bicubic weights
    for (int i = -1; i <= 2; ++i) {
        float weight_r = bicubic_kernel(r_frac - i);
        
        for (int j = -1; j <= 2; ++j) {
            float weight_c = bicubic_kernel(c_frac - j);
            float total_weight = weight_r * weight_c;
            
            // Only distribute if weight is non-zero
            if (total_weight != 0.0f) {
                accumulate_data_with_symmetry(data_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                            r_floor + i, c_floor + j, complex_scale(data_val, total_weight));
            }
        }
    }
}

// Distribute bilinear weights to 2x2 neighborhood
__device__ __forceinline__ void distribute_bilinear_weights(
    float* weight_rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    float weight_val, float r, float c
) {
    // Extract integer and fractional parts
    int c_floor = floorf(c);
    int r_floor = floorf(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute weights to 2x2 neighborhood with bilinear weights
    accumulate_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                   r_floor,     c_floor,     weight_val * (1.0f - r_frac) * (1.0f - c_frac));
    accumulate_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                   r_floor,     c_floor + 1, weight_val * (1.0f - r_frac) * c_frac);
    accumulate_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                   r_floor + 1, c_floor,     weight_val * r_frac * (1.0f - c_frac));
    accumulate_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                   r_floor + 1, c_floor + 1, weight_val * r_frac * c_frac);
}

// Distribute bicubic weights to 4x4 neighborhood
__device__ __forceinline__ void distribute_bicubic_weights(
    float* weight_rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    float weight_val, float r, float c
) {
    // Extract integer and fractional parts
    int c_floor = floorf(c);
    int r_floor = floorf(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute weights to 4x4 neighborhood using bicubic weights
    for (int i = -1; i <= 2; ++i) {
        float weight_r = bicubic_kernel(r_frac - i);
        
        for (int j = -1; j <= 2; ++j) {
            float weight_c = bicubic_kernel(c_frac - j);
            float total_weight = weight_r * weight_c;
            
            // Only distribute if weight is non-zero
            if (total_weight != 0.0f) {
                // For weights, we only need the interpolation weight magnitude (absolute value)
                accumulate_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                                r_floor + i, c_floor + j, weight_val * fabsf(total_weight));
            }
        }
    }
}

// Sample from FFTW-formatted Fourier space with automatic Friedel symmetry handling
__device__ __forceinline__ cuFloatComplex sample_fftw_with_conjugate(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    int r, int c
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
    int idx = b * rec_batch_stride + r * rec_row_stride + c;
    cuFloatComplex value = rec[idx];
    
    // Return conjugated value if we used Friedel symmetry
    return need_conjugate ? complex_conj(value) : value;
}

// Bilinear interpolation kernel
__device__ __forceinline__ cuFloatComplex bilinear_interpolate(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    float r, float c
) {
    // Extract integer and fractional parts
    int c_floor = floorf(c);
    int r_floor = floorf(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Sample 2x2 grid of neighboring pixels
    cuFloatComplex p00 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor, c_floor);
    cuFloatComplex p01 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor, c_floor + 1);
    cuFloatComplex p10 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor + 1, c_floor);
    cuFloatComplex p11 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor + 1, c_floor + 1);
    
    // Bilinear interpolation
    cuFloatComplex p0 = complex_add(p00, complex_scale(cuCsubf(p01, p00), c_frac));
    cuFloatComplex p1 = complex_add(p10, complex_scale(cuCsubf(p11, p10), c_frac));
    return complex_add(p0, complex_scale(cuCsubf(p1, p0), r_frac));
}

// Bilinear interpolation with gradients for backward pass
__device__ __forceinline__ void bilinear_interpolate_with_gradients(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    float r, float c,
    cuFloatComplex* val, cuFloatComplex* grad_r, cuFloatComplex* grad_c
) {
    // Extract integer and fractional parts
    int c_floor = floorf(c);
    int r_floor = floorf(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Sample 2x2 grid of neighboring pixels
    cuFloatComplex p00 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor, c_floor);
    cuFloatComplex p01 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor, c_floor + 1);
    cuFloatComplex p10 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor + 1, c_floor);
    cuFloatComplex p11 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r_floor + 1, c_floor + 1);
    
    // Value computation (same as interpolate method)
    cuFloatComplex p0 = complex_add(p00, complex_scale(cuCsubf(p01, p00), c_frac));
    cuFloatComplex p1 = complex_add(p10, complex_scale(cuCsubf(p11, p10), c_frac));
    *val = complex_add(p0, complex_scale(cuCsubf(p1, p0), r_frac));
    
    // Analytical spatial gradients derived from bilinear formula
    *grad_r = complex_add(complex_scale(cuCsubf(p10, p00), 1.0f - c_frac), 
                         complex_scale(cuCsubf(p11, p01), c_frac));
    *grad_c = complex_add(complex_scale(cuCsubf(p01, p00), 1.0f - r_frac), 
                         complex_scale(cuCsubf(p11, p10), r_frac));
}

// Safe sampling with edge clamping for bicubic interpolation
__device__ __forceinline__ cuFloatComplex sample_with_edge_clamping(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    int r, int c
) {
    // Clamp coordinates to valid ranges
    if (abs(c) >= boxsize_half) {
        c = (c < 0) ? -(boxsize_half - 1) : (boxsize_half - 1);
    }
    r = max(-boxsize / 2 + 1, min(r, boxsize / 2));
    
    return sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride, r, c);
}

// Bicubic interpolation kernel
__device__ __forceinline__ cuFloatComplex bicubic_interpolate(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    float r, float c
) {
    // Extract integer and fractional parts
    int c_floor = floorf(c);
    int r_floor = floorf(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    cuFloatComplex result = make_cuFloatComplex(0.0f, 0.0f);
    
    // Sample 4x4 grid around the interpolation point
    for (int i = -1; i <= 2; ++i) {
        float weight_r = bicubic_kernel(r_frac - i);
        
        for (int j = -1; j <= 2; ++j) {
            cuFloatComplex sample = sample_with_edge_clamping(rec, b, boxsize, boxsize_half, 
                                                            rec_batch_stride, rec_row_stride,
                                                            r_floor + i, c_floor + j);
            float weight_c = bicubic_kernel(c_frac - j);
            result = complex_add(result, complex_scale(sample, weight_r * weight_c));
        }
    }
    
    return result;
}

// Bicubic kernel derivative for gradient computation
__device__ __forceinline__ float bicubic_kernel_derivative(float s) {
    const float a = -0.5f;
    float sign = (s < 0) ? -1.0f : 1.0f;
    s = fabsf(s);
    
    if (s <= 1.0f) {
        return sign * (3.0f * (a + 2.0f) * s * s - 2.0f * (a + 3.0f) * s);
    } else if (s <= 2.0f) {
        return sign * (3.0f * a * s * s - 10.0f * a * s + 8.0f * a);
    } else {
        return 0.0f;
    }
}

// Bicubic interpolation with gradients for backward pass
__device__ __forceinline__ void bicubic_interpolate_with_gradients(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    float r, float c,
    cuFloatComplex* val, cuFloatComplex* grad_r, cuFloatComplex* grad_c
) {
    // Extract integer and fractional parts
    int c_floor = floorf(c);
    int r_floor = floorf(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    *val = make_cuFloatComplex(0.0f, 0.0f);
    *grad_r = make_cuFloatComplex(0.0f, 0.0f);
    *grad_c = make_cuFloatComplex(0.0f, 0.0f);
    
    // Sample 4x4 grid and compute value + gradients simultaneously
    for (int i = -1; i <= 2; ++i) {
        float weight_r = bicubic_kernel(r_frac - i);
        float dweight_r = bicubic_kernel_derivative(r_frac - i);
        
        for (int j = -1; j <= 2; ++j) {
            cuFloatComplex sample = sample_with_edge_clamping(rec, b, boxsize, boxsize_half, 
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

// Forward backproject 2D kernel - accumulates 2D projections into 2D reconstructions
__global__ void backproject_2d_forw_kernel(
    const cuFloatComplex* projections,
    const float* weights,
    const float* rotations,
    const float* shifts,
    cuFloatComplex* data_reconstruction,
    float* weight_reconstruction,
    CudaParams params
) {
    // Thread organization: each block handles one entire projection
    // Grid: (P, B, 1), Block: (256, 1, 1)
    int p = blockIdx.x;  // Pose index
    int b = blockIdx.y;  // Batch index
    
    // Check bounds for batch and pose
    if (p >= params.P || b >= params.B) {
        return;
    }
    
    // Pre-compute shared parameters for this projection (pose p, batch b)
    
    // Rotation matrix (shared for all pixels in this projection)
    int rot_b_idx = (params.B_rot == 1) ? 0 : b;
    int rot_idx_base = (rot_b_idx * params.P + p) * 4;  // Index to start of 2x2 matrix
    float rot_00 = rotations[rot_idx_base + 0];     // R[0][0]
    float rot_01 = rotations[rot_idx_base + 1];     // R[0][1] 
    float rot_10 = rotations[rot_idx_base + 2];     // R[1][0]
    float rot_11 = rotations[rot_idx_base + 3];     // R[1][1]
    
    // Shifts (shared for all pixels in this projection)
    float shift_r = 0.0f, shift_c = 0.0f;
    if (params.has_shifts) {
        int shift_b_idx = (params.B_shift == 1) ? 0 : b;
        int shift_idx_base = (shift_b_idx * params.P + p) * 2;
        shift_r = shifts[shift_idx_base + 0];
        shift_c = shifts[shift_idx_base + 1];
    }
    
    // Pre-compute constants
    int proj_batch_stride = params.P * params.proj_boxsize * params.proj_boxsize_half;
    int proj_pose_stride = params.proj_boxsize * params.proj_boxsize_half;
    int proj_base_idx = b * proj_batch_stride + p * proj_pose_stride;
    
    int rec_batch_stride = params.boxsize * params.boxsize_half;
    int rec_row_stride = params.boxsize_half;
    
    float fourier_cutoff_sq = params.fourier_radius_cutoff * params.fourier_radius_cutoff;
    
    bool has_weights = (weights != nullptr);
    
    // Loop over all pixels in this projection
    int total_pixels = params.proj_boxsize * params.proj_boxsize_half;
    
    for (int pixel_idx = threadIdx.x; pixel_idx < total_pixels; pixel_idx += blockDim.x) {
        // Convert linear pixel index to (i, j) coordinates
        int i = pixel_idx / params.proj_boxsize_half;  // Row
        int j = pixel_idx % params.proj_boxsize_half;  // Column
        
        // Convert array indices to Fourier coordinates (same as CPU)
        float proj_coord_c = float(j);  // Column: always positive (FFTW half-space)
        float proj_coord_r = (i <= params.proj_boxsize / 2) ? float(i) : float(i) - float(params.proj_boxsize);
        
        // Apply Fourier space filtering
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            continue;
        }
        
        // Skip Friedel-symmetric half of the x = 0 line (handled by other half)
        if (j == 0 && i >= params.proj_boxsize / 2) {
            continue;
        }
        
        // Apply oversampling scaling to coordinates
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Apply rotation matrix (using pre-computed matrix elements)
        float rot_c = rot_00 * sample_c + rot_01 * sample_r;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r;
        
        // Get projection data
        cuFloatComplex proj_val = projections[proj_base_idx + pixel_idx];
        
        // Apply conjugate phase shift for back-projection
        if (params.has_shifts) {
            float phase = -2.0f * M_PI * (proj_coord_r * shift_r / params.boxsize + 
                                          proj_coord_c * shift_c / params.boxsize);
            cuFloatComplex phase_factor = make_cuFloatComplex(cosf(phase), sinf(phase));
            proj_val = complex_mul(proj_val, complex_conj(phase_factor));  // conjugate for backprojection
        }
        
        // Distribute projection data to reconstruction 
        if (params.interpolation_method == 0) {  // linear
            distribute_bilinear_data(data_reconstruction, b, params.boxsize, params.boxsize_half,
                                   rec_batch_stride, rec_row_stride, proj_val, rot_r, rot_c);
        } else {  // cubic
            distribute_bicubic_data(data_reconstruction, b, params.boxsize, params.boxsize_half,
                                  rec_batch_stride, rec_row_stride, proj_val, rot_r, rot_c);
        }
        
        // Distribute weights if provided
        if (has_weights) {
            float weight_val = weights[proj_base_idx + pixel_idx];
            
            if (params.interpolation_method == 0) {  // linear
                distribute_bilinear_weights(weight_reconstruction, b, params.boxsize, params.boxsize_half,
                                          rec_batch_stride, rec_row_stride, weight_val, rot_r, rot_c);
            } else {  // cubic  
                distribute_bicubic_weights(weight_reconstruction, b, params.boxsize, params.boxsize_half,
                                         rec_batch_stride, rec_row_stride, weight_val, rot_r, rot_c);
            }
        }
    }
}

// Sample gradient from weight reconstruction with bounds and symmetry handling
__device__ __forceinline__ float sample_weight_gradient(
    const float* grad_weight_rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_row_stride,
    int r, int c
) {
    // Handle Friedel symmetry and bounds
    if (c < 0) { 
        c = -c;
        r = -r;
    }
    if (c >= boxsize_half) return 0.0f;
    if (r > boxsize / 2 || r < -boxsize / 2 + 1) return 0.0f;
    
    int r_eff = r < 0 ? boxsize + r : r;
    if (r_eff >= boxsize) return 0.0f;
    
    return grad_weight_rec[b * rec_batch_stride + r_eff * rec_row_stride + c];
}

// Backward backproject 2D kernel - computes gradients for backprojection operation
__global__ void backproject_2d_back_kernel(
    const cuFloatComplex* grad_data_rec,
    const float* grad_weight_rec,
    const cuFloatComplex* projections,
    const float* weights,
    const float* rotations,
    const float* shifts,
    cuFloatComplex* grad_projections,
    float* grad_weights,
    float* grad_rotations,
    float* grad_shifts,
    CudaParams params
) {
    // Thread organization: each block handles one entire projection
    // Grid: (P, B, 1), Block: (256, 1, 1)
    int p = blockIdx.x;  // Pose index
    int b = blockIdx.y;  // Batch index
    
    if (p >= params.P || b >= params.B) {
        return;
    }
    
    // Extract gradient flags from interpolation_method upper bits
    bool need_rotation_grads = (params.interpolation_method & 0x10) != 0;
    bool need_shift_grads = (params.interpolation_method & 0x20) != 0;
    int interpolation_method = params.interpolation_method & 0x0F;  // Lower 4 bits
    bool has_weights = (weights != nullptr);
    bool need_weight_grads = (grad_weight_rec != nullptr);
    
    // Local accumulators for gradients that need reduction
    __shared__ float local_rot_grad[256][4];    // [thread][matrix_element] 
    __shared__ float local_shift_grad[256][2];  // [thread][shift_component]
    
    int tid = threadIdx.x;
    
    // Initialize local accumulators
    local_rot_grad[tid][0] = 0.0f;   // R[0][0]
    local_rot_grad[tid][1] = 0.0f;   // R[0][1] 
    local_rot_grad[tid][2] = 0.0f;   // R[1][0]
    local_rot_grad[tid][3] = 0.0f;   // R[1][1]
    local_shift_grad[tid][0] = 0.0f; // shift_r
    local_shift_grad[tid][1] = 0.0f; // shift_c
    
    // Pre-compute shared parameters for this projection
    int rot_b_idx = (params.B_rot == 1) ? 0 : b;
    int rot_idx_base = (rot_b_idx * params.P + p) * 4;
    float rot_00 = rotations[rot_idx_base + 0];
    float rot_01 = rotations[rot_idx_base + 1];
    float rot_10 = rotations[rot_idx_base + 2];
    float rot_11 = rotations[rot_idx_base + 3];
    
    float shift_r = 0.0f, shift_c = 0.0f;
    if (params.has_shifts) {
        int shift_b_idx = (params.B_shift == 1) ? 0 : b;
        int shift_idx_base = (shift_b_idx * params.P + p) * 2;
        shift_r = shifts[shift_idx_base + 0];
        shift_c = shifts[shift_idx_base + 1];
    }
    
    // Pre-compute strides
    int proj_batch_stride = params.P * params.proj_boxsize * params.proj_boxsize_half;
    int proj_pose_stride = params.proj_boxsize * params.proj_boxsize_half;
    int proj_base_idx = b * proj_batch_stride + p * proj_pose_stride;
    
    int rec_batch_stride = params.boxsize * params.boxsize_half;
    int rec_row_stride = params.boxsize_half;
    
    float fourier_cutoff_sq = params.fourier_radius_cutoff * params.fourier_radius_cutoff;
    
    // Loop over all pixels in this projection
    int total_pixels = params.proj_boxsize * params.proj_boxsize_half;
    
    for (int pixel_idx = tid; pixel_idx < total_pixels; pixel_idx += blockDim.x) {
        int i = pixel_idx / params.proj_boxsize_half;
        int j = pixel_idx % params.proj_boxsize_half;
        
        // Convert array indices to Fourier coordinates
        float proj_coord_c = float(j);
        float proj_coord_r = (i <= params.proj_boxsize / 2) ? float(i) : float(i) - float(params.proj_boxsize);
        
        // Apply Fourier space filtering
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            continue;
        }
        
        if (j == 0 && i >= params.proj_boxsize / 2) {
            // Skip Friedel-symmetric half of the x = 0 line
            continue;
        }
        
        // Apply oversampling scaling
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Apply rotation matrix
        float rot_c = rot_00 * sample_c + rot_01 * sample_r;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r;
        
        // 1. Compute grad_projections using forward projection (adjoint relationship)
        cuFloatComplex rec_val;
        if (interpolation_method == 0) {  // linear
            rec_val = bilinear_interpolate(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                         rec_batch_stride, rec_row_stride, rot_r, rot_c);
        } else {  // cubic
            rec_val = bicubic_interpolate(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                        rec_batch_stride, rec_row_stride, rot_r, rot_c);
        }
        
        // Apply conjugate phase shift (opposite of back-projection)
        if (params.has_shifts) {
            float phase = -2.0f * M_PI * (proj_coord_r * shift_r / params.boxsize + 
                                          proj_coord_c * shift_c / params.boxsize);
            cuFloatComplex phase_factor = make_cuFloatComplex(cosf(phase), sinf(phase));
            rec_val = complex_mul(rec_val, phase_factor);  // forward phase for grad_projections
        }
        
        grad_projections[proj_base_idx + pixel_idx] = rec_val;
        
        // 2. Compute grad_weights if needed
        if (need_weight_grads && has_weights) {
            // Use linear interpolation for real-valued gradient
            int c_floor = floorf(rot_c);
            int r_floor = floorf(rot_r);
            float c_frac = rot_c - c_floor;
            float r_frac = rot_r - r_floor;
            
            // Sample 2x2 grid from grad_weight_rec with bounds checking
            const float p00 = sample_weight_gradient(grad_weight_rec, b, params.boxsize, params.boxsize_half, 
                                                    rec_batch_stride, rec_row_stride, r_floor, c_floor);
            const float p01 = sample_weight_gradient(grad_weight_rec, b, params.boxsize, params.boxsize_half, 
                                                    rec_batch_stride, rec_row_stride, r_floor, c_floor + 1);
            const float p10 = sample_weight_gradient(grad_weight_rec, b, params.boxsize, params.boxsize_half, 
                                                    rec_batch_stride, rec_row_stride, r_floor + 1, c_floor);
            const float p11 = sample_weight_gradient(grad_weight_rec, b, params.boxsize, params.boxsize_half, 
                                                    rec_batch_stride, rec_row_stride, r_floor + 1, c_floor + 1);
            
            // Bilinear interpolation
            const float p0 = p00 + (p01 - p00) * c_frac;
            const float p1 = p10 + (p11 - p10) * c_frac;
            const float weight_grad = p0 + (p1 - p0) * r_frac;
            
            grad_weights[proj_base_idx + pixel_idx] = weight_grad;
        }
        
        // 3. Compute rotation gradients if needed
        if (need_rotation_grads) {
            cuFloatComplex _unused, grad_r, grad_c;
            if (interpolation_method == 0) {  // linear
                bilinear_interpolate_with_gradients(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                                  rec_batch_stride, rec_row_stride, rot_r, rot_c,
                                                  &_unused, &grad_r, &grad_c);
            } else {  // cubic
                bicubic_interpolate_with_gradients(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                                 rec_batch_stride, rec_row_stride, rot_r, rot_c,
                                                 &_unused, &grad_r, &grad_c);
            }
            
            cuFloatComplex proj_val = projections[proj_base_idx + pixel_idx];
            
            // Apply conjugate phase shift to projection value
            if (params.has_shifts) {
                float phase = -2.0f * M_PI * (proj_coord_r * shift_r / params.boxsize + 
                                              proj_coord_c * shift_c / params.boxsize);
                cuFloatComplex phase_factor = make_cuFloatComplex(cosf(phase), sinf(phase));
                proj_val = complex_mul(proj_val, complex_conj(phase_factor));
            }
            
            // Chain rule for rotation matrix gradients
            local_rot_grad[tid][0] += cuCrealf(complex_mul(proj_val, complex_conj(complex_scale(grad_c, sample_c))));
            local_rot_grad[tid][1] += cuCrealf(complex_mul(proj_val, complex_conj(complex_scale(grad_c, sample_r))));
            local_rot_grad[tid][2] += cuCrealf(complex_mul(proj_val, complex_conj(complex_scale(grad_r, sample_c))));
            local_rot_grad[tid][3] += cuCrealf(complex_mul(proj_val, complex_conj(complex_scale(grad_r, sample_r))));
        }
        
        // 4. Compute shift gradients if needed
        if (need_shift_grads) {
            cuFloatComplex rec_val_for_shift;
            if (interpolation_method == 0) {  // linear
                rec_val_for_shift = bilinear_interpolate(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                                       rec_batch_stride, rec_row_stride, rot_r, rot_c);
            } else {  // cubic
                rec_val_for_shift = bicubic_interpolate(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                                      rec_batch_stride, rec_row_stride, rot_r, rot_c);
            }
            
            cuFloatComplex proj_val = projections[proj_base_idx + pixel_idx];
            
            // Apply conjugate phase shift to projection value (consistent with rotation gradient computation)
            if (params.has_shifts) {
                float phase = -2.0f * M_PI * (proj_coord_r * shift_r / params.boxsize + 
                                              proj_coord_c * shift_c / params.boxsize);
                cuFloatComplex phase_factor = make_cuFloatComplex(cosf(phase), sinf(phase));
                proj_val = complex_mul(proj_val, complex_conj(phase_factor));
            }
            
            cuFloatComplex phase_grad_r = complex_mul(make_cuFloatComplex(0.0f, -2.0f * M_PI * proj_coord_r / params.boxsize), rec_val_for_shift);
            cuFloatComplex phase_grad_c = complex_mul(make_cuFloatComplex(0.0f, -2.0f * M_PI * proj_coord_c / params.boxsize), rec_val_for_shift);
            
            local_shift_grad[tid][0] += cuCrealf(complex_mul(proj_val, complex_conj(phase_grad_r)));
            local_shift_grad[tid][1] += cuCrealf(complex_mul(proj_val, complex_conj(phase_grad_c)));
        }
    }
    
    __syncthreads();
    
    // Reduce and write gradients (only thread 0 writes to avoid races)
    if (tid == 0) {
        // Reduce rotation gradients across threads
        if (need_rotation_grads) {
            float total_rot_grad[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int t = 0; t < blockDim.x; ++t) {
                total_rot_grad[0] += local_rot_grad[t][0];
                total_rot_grad[1] += local_rot_grad[t][1];
                total_rot_grad[2] += local_rot_grad[t][2];
                total_rot_grad[3] += local_rot_grad[t][3];
            }
            
            // Atomically add to global rotation gradients
            int global_rot_idx_base = rot_idx_base;
            atomic_add_real(&grad_rotations[global_rot_idx_base + 0], total_rot_grad[0]);
            atomic_add_real(&grad_rotations[global_rot_idx_base + 1], total_rot_grad[1]);
            atomic_add_real(&grad_rotations[global_rot_idx_base + 2], total_rot_grad[2]);
            atomic_add_real(&grad_rotations[global_rot_idx_base + 3], total_rot_grad[3]);
        }
        
        // Reduce shift gradients across threads
        if (need_shift_grads) {
            float total_shift_grad[2] = {0.0f, 0.0f};
            for (int t = 0; t < blockDim.x; ++t) {
                total_shift_grad[0] += local_shift_grad[t][0];
                total_shift_grad[1] += local_shift_grad[t][1];
            }
            
            // Atomically add to global shift gradients
            int shift_b_idx = (params.B_shift == 1) ? 0 : b;
            int global_shift_idx_base = (shift_b_idx * params.P + p) * 2;
            atomic_add_real(&grad_shifts[global_shift_idx_base + 0], total_shift_grad[0]);
            atomic_add_real(&grad_shifts[global_shift_idx_base + 1], total_shift_grad[1]);
        }
    }
}

// Forward backprojection from 2D projections to 2D reconstructions (CUDA version)
std::tuple<at::Tensor, at::Tensor> backproject_2d_forw_cuda(
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Input validation
    TORCH_CHECK(projections.is_cuda(), "Input projections must be on CUDA device");
    TORCH_CHECK(rotations.is_cuda(), "Input rotations must be on CUDA device");
    TORCH_CHECK(projections.is_complex(), "Projections must be a complex tensor");
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic",
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(projections.dim() == 4,
                "Projections must be a 4D tensor (B, P, height, width/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2,
                "Rotations must be (B_rot, P, 2, 2)");

    // Validate optional weights
    if (weights.has_value()) {
        TORCH_CHECK(weights->is_cuda(), "Weights must be on CUDA device");
        TORCH_CHECK(weights->is_floating_point(), "Weights must be a real-valued tensor");
        TORCH_CHECK(weights->dim() == 4, "Weights must be (B, P, height, width/2+1)");
        TORCH_CHECK(weights->sizes() == projections.sizes(), "Weights and projections must have the same shape");
    }

    // Extract tensor dimensions
    const auto B = projections.size(0);
    const auto P = projections.size(1); 
    const auto proj_boxsize = projections.size(2);
    const auto proj_boxsize_half = projections.size(3);
    
    const auto B_rot = rotations.size(0);
    TORCH_CHECK(B_rot == B || B_rot == 1, "Batch size of rotations must be 1 or same as projections");

    // Reconstruction size matches projection size
    const auto rec_boxsize = proj_boxsize;
    const auto rec_boxsize_half = proj_boxsize_half;
    
    // Initialize output tensors
    auto data_reconstruction = torch::zeros({B, rec_boxsize, rec_boxsize_half}, projections.options());
    at::Tensor weight_reconstruction;
    bool has_weights = weights.has_value();
    
    if (has_weights) {
        weight_reconstruction = torch::zeros({B, rec_boxsize, rec_boxsize_half}, weights->options());
    } else {
        weight_reconstruction = torch::empty({0}, projections.options().dtype(rotations.scalar_type()));
    }

    // Set CUDA device guard to ensure operations happen on the right device
    const c10::cuda::CUDAGuard device_guard(projections.device());
    
    // Get CUDA stream for asynchronous operations
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Validate shifts tensor if provided
    c10::optional<at::Tensor> shifts_contiguous;
    int64_t B_shift = 1;
    if (shifts.has_value()) {
        TORCH_CHECK(shifts->is_cuda(), "Shifts must be on CUDA device");
        TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
        TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1, "Batch size of shifts must be 1 or same as projections");
        TORCH_CHECK(shifts->size(1) == P, "Number of poses in shifts must match rotations");
        TORCH_CHECK(shifts->scalar_type() == rotations.scalar_type(), "Shifts and rotations must have the same dtype");
        B_shift = shifts->size(0);
        shifts_contiguous = shifts->is_contiguous() ? *shifts : shifts->contiguous();
    }

    // Ensure tensors are contiguous
    auto proj_contiguous = projections.is_contiguous() ? projections : projections.contiguous();
    auto rot_contiguous = rotations.is_contiguous() ? rotations : rotations.contiguous();
    auto data_rec_contiguous = data_reconstruction.is_contiguous() ? data_reconstruction : data_reconstruction.contiguous();
    c10::optional<at::Tensor> weights_contiguous;
    c10::optional<at::Tensor> weight_rec_contiguous;
    if (has_weights) {
        weights_contiguous = weights->is_contiguous() ? *weights : weights->contiguous();
        weight_rec_contiguous = weight_reconstruction.is_contiguous() ? weight_reconstruction : weight_reconstruction.contiguous();
    }

    // Set up kernel parameters
    CudaParams params = {
        (int)B, (int)P, (int)rec_boxsize, (int)rec_boxsize_half,
        (int)proj_boxsize, (int)proj_boxsize_half, (int)B_rot, (int)B_shift,
        (int)(shifts.has_value() ? 1 : 0),
        (interpolation == "linear") ? 0 : 1,
        static_cast<float>(oversampling),
        static_cast<float>(fourier_radius_cutoff.value_or(proj_boxsize / 2.0))
    };

    // Launch CUDA kernel
    // Grid: (P, B, 1), Block: (256, 1, 1) - each block handles one projection
    dim3 gridDim(P, B, 1);
    dim3 blockDim(256, 1, 1);

    // Check if we need double precision - fall back to CPU for gradcheck
    if (projections.scalar_type() == at::kComplexDouble) {
        // For double precision (used by gradcheck), fall back to CPU since we don't 
        // want to implement full double precision CUDA kernels just for testing
        auto projections_cpu = projections.cpu();
        auto rotations_cpu = rotations.cpu();
        c10::optional<at::Tensor> weights_cpu;
        c10::optional<at::Tensor> shifts_cpu;
        if (weights.has_value()) {
            weights_cpu = weights->cpu();
        }
        if (shifts.has_value()) {
            shifts_cpu = shifts->cpu();
        }
        
        auto [data_rec_cpu, weight_rec_cpu] = backproject_2d_forw_cpu(
            projections_cpu, weights_cpu, rotations_cpu, shifts_cpu,
            interpolation, oversampling, fourier_radius_cutoff
        );
        
        auto device = projections.device();
        auto data_result = data_rec_cpu.to(device, /*non_blocking=*/false);
        auto weight_result = weight_rec_cpu.numel() > 0 ? weight_rec_cpu.to(device, /*non_blocking=*/false) : weight_rec_cpu;
        
        return std::make_tuple(data_result, weight_result);
    }

    // Get raw pointers for kernel launch (float32 only)
    const cuFloatComplex* proj_ptr = reinterpret_cast<const cuFloatComplex*>(proj_contiguous.data_ptr<c10::complex<float>>());
    const float* rot_ptr = rot_contiguous.data_ptr<float>();
    const float* weights_ptr = has_weights ? weights_contiguous->data_ptr<float>() : nullptr;
    const float* shift_ptr = shifts.has_value() ? shifts_contiguous->data_ptr<float>() : nullptr;
    cuFloatComplex* data_rec_ptr = reinterpret_cast<cuFloatComplex*>(data_rec_contiguous.data_ptr<c10::complex<float>>());
    float* weight_rec_ptr = has_weights ? weight_rec_contiguous->data_ptr<float>() : nullptr;

    backproject_2d_forw_kernel<<<gridDim, blockDim, 0, stream>>>(
        proj_ptr, weights_ptr, rot_ptr, shift_ptr, data_rec_ptr, weight_rec_ptr, params
    );

    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Copy back if needed
    if (!data_reconstruction.is_contiguous()) {
        data_reconstruction.copy_(data_rec_contiguous);
    }
    if (has_weights && !weight_reconstruction.is_contiguous()) {
        weight_reconstruction.copy_(*weight_rec_contiguous);
    }
    
    return std::make_tuple(data_reconstruction, weight_reconstruction);
}

// Backward backprojection for gradients (CUDA version) 
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backproject_2d_back_cuda(
    const at::Tensor& grad_data_rec,
    const c10::optional<at::Tensor>& grad_weight_rec,
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Input validation
    TORCH_CHECK(grad_data_rec.is_cuda(), "Input grad_data_rec must be on CUDA device");
    TORCH_CHECK(projections.is_cuda(), "Input projections must be on CUDA device");
    TORCH_CHECK(rotations.is_cuda(), "Input rotations must be on CUDA device");
    TORCH_CHECK(grad_data_rec.is_complex(), "grad_data_rec must be a complex tensor");
    TORCH_CHECK(projections.is_complex(), "projections must be a complex tensor");
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic",
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(grad_data_rec.dim() == 3, "grad_data_rec must be (B, height, width/2+1)");
    TORCH_CHECK(projections.dim() == 4, "projections must be a 4D tensor (B, P, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2,
                "Rotations must be (B_rot, P, 2, 2)");
    
    if (grad_weight_rec.has_value()) {
        TORCH_CHECK(grad_weight_rec->is_cuda(), "grad_weight_rec must be on CUDA device");
        TORCH_CHECK(grad_weight_rec->is_floating_point(), "grad_weight_rec must be a real-valued tensor");
        TORCH_CHECK(grad_weight_rec->dim() == 3, "grad_weight_rec must be (B, height, width/2+1)");
        TORCH_CHECK(grad_weight_rec->sizes() == grad_data_rec.sizes(), "grad_weight_rec and grad_data_rec must have same shape");
    }

    const auto B = projections.size(0);
    const auto P = projections.size(1);
    const auto proj_boxsize = projections.size(2);
    const auto proj_boxsize_half = projections.size(3);
    
    const auto rec_boxsize = grad_data_rec.size(1);
    const auto rec_boxsize_half = grad_data_rec.size(2);
    
    const auto B_rot = rotations.size(0);
    TORCH_CHECK(B_rot == B || B_rot == 1, "Batch size of rotations must be 1 or same as projections");

    // Initialize gradient tensors based on what's needed
    auto grad_projections = torch::zeros_like(projections);
    
    at::Tensor grad_weights;
    at::Tensor grad_rotations;
    at::Tensor grad_shifts;
    
    const bool need_rotation_grads = rotations.requires_grad();
    const bool need_shift_grads = shifts.has_value() && shifts->requires_grad();
    const bool has_weights = weights.has_value();
    
    if (has_weights) {
        TORCH_CHECK(weights->is_cuda(), "weights must be on CUDA device");
        grad_weights = torch::zeros_like(*weights);
    } else {
        grad_weights = torch::empty({0}, projections.options().dtype(rotations.scalar_type()));
    }
    
    if (need_rotation_grads) {
        grad_rotations = torch::zeros_like(rotations);
    } else {
        grad_rotations = torch::empty({0}, rotations.options());
    }

    int64_t B_shift = 1;
    if (shifts.has_value()) {
        TORCH_CHECK(shifts->is_cuda(), "Shifts must be on CUDA device");
        TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
        TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1, "Batch size of shifts must be 1 or same as projections");
        TORCH_CHECK(shifts->size(1) == P, "Number of poses in shifts must match rotations");
        TORCH_CHECK(shifts->scalar_type() == rotations.scalar_type(), "Shifts and rotations must have the same dtype");
        B_shift = shifts->size(0);
        
        if (need_shift_grads) {
            grad_shifts = torch::zeros_like(*shifts);
        } else {
            grad_shifts = torch::empty({0}, projections.options().dtype(rotations.scalar_type()));
        }
    } else {
        grad_shifts = torch::empty({0}, projections.options().dtype(rotations.scalar_type()));
    }

    // Set CUDA device guard
    const c10::cuda::CUDAGuard device_guard(grad_data_rec.device());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Ensure tensors are contiguous
    auto grad_data_rec_contiguous = grad_data_rec.is_contiguous() ? grad_data_rec : grad_data_rec.contiguous();
    auto proj_contiguous = projections.is_contiguous() ? projections : projections.contiguous();
    auto rot_contiguous = rotations.is_contiguous() ? rotations : rotations.contiguous();
    auto grad_proj_contiguous = grad_projections.is_contiguous() ? grad_projections : grad_projections.contiguous();

    c10::optional<at::Tensor> grad_weight_rec_contiguous;
    c10::optional<at::Tensor> weights_contiguous;
    c10::optional<at::Tensor> grad_weights_contiguous;
    c10::optional<at::Tensor> shifts_contiguous;
    c10::optional<at::Tensor> grad_shifts_contiguous;
    c10::optional<at::Tensor> grad_rot_contiguous;

    if (grad_weight_rec.has_value()) {
        grad_weight_rec_contiguous = grad_weight_rec->is_contiguous() ? *grad_weight_rec : grad_weight_rec->contiguous();
    }
    if (has_weights) {
        weights_contiguous = weights->is_contiguous() ? *weights : weights->contiguous();
        grad_weights_contiguous = grad_weights.is_contiguous() ? grad_weights : grad_weights.contiguous();
    }
    if (shifts.has_value()) {
        shifts_contiguous = shifts->is_contiguous() ? *shifts : shifts->contiguous();
        if (need_shift_grads) {
            grad_shifts_contiguous = grad_shifts.is_contiguous() ? grad_shifts : grad_shifts.contiguous();
        }
    }
    if (need_rotation_grads) {
        grad_rot_contiguous = grad_rotations.is_contiguous() ? grad_rotations : grad_rotations.contiguous();
    }

    // Set up kernel parameters with gradient flags
    CudaParams params = {
        (int)B, (int)P, (int)rec_boxsize, (int)rec_boxsize_half,
        (int)proj_boxsize, (int)proj_boxsize_half, (int)B_rot, (int)B_shift,
        (int)(shifts.has_value() ? 1 : 0),
        (interpolation == "linear" ? 0 : 1) | 
        (need_rotation_grads ? 0x10 : 0) | 
        (need_shift_grads ? 0x20 : 0),  // Pack flags into upper bits
        static_cast<float>(oversampling),
        static_cast<float>(fourier_radius_cutoff.value_or(proj_boxsize / 2.0))
    };

    // Launch CUDA kernel
    dim3 gridDim(P, B, 1);
    dim3 blockDim(256, 1, 1);

    // Check if we need double precision - fall back to CPU for gradcheck
    if (grad_data_rec.scalar_type() == at::kComplexDouble) {
        // For double precision (used by gradcheck), fall back to CPU
        auto grad_data_rec_cpu = grad_data_rec.cpu();
        c10::optional<at::Tensor> grad_weight_rec_cpu;
        auto projections_cpu = projections.cpu();
        c10::optional<at::Tensor> weights_cpu;
        auto rotations_cpu = rotations.cpu();
        c10::optional<at::Tensor> shifts_cpu;
        
        if (grad_weight_rec.has_value()) {
            grad_weight_rec_cpu = grad_weight_rec->cpu();
        }
        if (weights.has_value()) {
            weights_cpu = weights->cpu();
        }
        if (shifts.has_value()) {
            shifts_cpu = shifts->cpu();
        }
        
        auto [grad_proj_cpu, grad_weights_cpu, grad_rot_cpu, grad_shifts_cpu] = backproject_2d_back_cpu(
            grad_data_rec_cpu, grad_weight_rec_cpu, projections_cpu, weights_cpu, 
            rotations_cpu, shifts_cpu, interpolation, oversampling, fourier_radius_cutoff
        );
        
        auto device = grad_data_rec.device();
        auto grad_proj_result = grad_proj_cpu.to(device, /*non_blocking=*/false);
        auto grad_weights_result = grad_weights_cpu.numel() > 0 ? grad_weights_cpu.to(device, /*non_blocking=*/false) : grad_weights_cpu;
        auto grad_rot_result = grad_rot_cpu.numel() > 0 ? grad_rot_cpu.to(device, /*non_blocking=*/false) : grad_rot_cpu;
        auto grad_shift_result = grad_shifts_cpu.numel() > 0 ? grad_shifts_cpu.to(device, /*non_blocking=*/false) : grad_shifts_cpu;
        
        return std::make_tuple(grad_proj_result, grad_weights_result, grad_rot_result, grad_shift_result);
    }

    // Get raw pointers for kernel launch (float32 only)
    const cuFloatComplex* grad_data_rec_ptr = reinterpret_cast<const cuFloatComplex*>(grad_data_rec_contiguous.data_ptr<c10::complex<float>>());
    const float* grad_weight_rec_ptr = grad_weight_rec.has_value() ? grad_weight_rec_contiguous->data_ptr<float>() : nullptr;
    const cuFloatComplex* proj_ptr = reinterpret_cast<const cuFloatComplex*>(proj_contiguous.data_ptr<c10::complex<float>>());
    const float* weights_ptr = has_weights ? weights_contiguous->data_ptr<float>() : nullptr;
    const float* rot_ptr = rot_contiguous.data_ptr<float>();
    const float* shift_ptr = shifts.has_value() ? shifts_contiguous->data_ptr<float>() : nullptr;
    cuFloatComplex* grad_proj_ptr = reinterpret_cast<cuFloatComplex*>(grad_proj_contiguous.data_ptr<c10::complex<float>>());
    float* grad_weights_ptr = has_weights ? grad_weights_contiguous->data_ptr<float>() : nullptr;
    float* grad_rot_ptr = need_rotation_grads ? grad_rot_contiguous->data_ptr<float>() : nullptr;
    float* grad_shift_ptr = need_shift_grads ? grad_shifts_contiguous->data_ptr<float>() : nullptr;

    backproject_2d_back_kernel<<<gridDim, blockDim, 0, stream>>>(
        grad_data_rec_ptr, grad_weight_rec_ptr, proj_ptr, weights_ptr, rot_ptr, shift_ptr,
        grad_proj_ptr, grad_weights_ptr, grad_rot_ptr, grad_shift_ptr, params
    );

    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Copy back if needed
    if (!grad_projections.is_contiguous()) {
        grad_projections.copy_(grad_proj_contiguous);
    }
    if (has_weights && !grad_weights.is_contiguous()) {
        grad_weights.copy_(*grad_weights_contiguous);
    }
    if (need_rotation_grads && !grad_rotations.is_contiguous()) {
        grad_rotations.copy_(*grad_rot_contiguous);
    }
    if (need_shift_grads && !grad_shifts.is_contiguous()) {
        grad_shifts.copy_(*grad_shifts_contiguous);
    }

    return std::make_tuple(grad_projections, grad_weights, grad_rotations, grad_shifts);
}

#endif // USE_CUDA