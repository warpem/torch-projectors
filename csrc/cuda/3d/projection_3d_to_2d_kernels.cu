#include "projection_3d_to_2d_kernels.h"

#ifdef USE_CUDA

#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../../cpu/3d/projection_3d_to_2d_kernels.h"

// CUDA utility functions and kernels for 3D->2D projection

struct CudaParams3D {
    int B, P, boxsize, boxsize_half;
    int proj_boxsize, proj_boxsize_half, B_rot, B_shift;
    int has_shifts;
    int interpolation_method; // 0=linear, 1=cubic
    float oversampling;
    float fourier_radius_cutoff;
    // Texture objects for B=1 optimization
    cudaTextureObject_t tex_real;
    cudaTextureObject_t tex_imag;
    bool use_textures;
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

// Sample from 3D volume using texture objects (when available)
__device__ __forceinline__ cuFloatComplex sample_3d_texture(
    cudaTextureObject_t tex_real, cudaTextureObject_t tex_imag,
    int boxsize, int boxsize_half,
    int d, int r, int c
) {
    bool need_conjugate = false;
    
    // Handle negative kx via 3D Friedel symmetry (c < 0)
    if (c < 0) {
        c = -c;          // Mirror to positive kx
        r = -r;          // ky must be mirrored as well
        d = -d;          // kz must be mirrored as well
        need_conjugate = !need_conjugate;
    }
    
    // Clamp coordinates to valid array bounds
    c = min(c, boxsize_half - 1);  // Column: [0, boxsize/2]
    
    // Row and depth: [-boxsize/2+1, boxsize/2]
    r = min(boxsize / 2, max(r, -boxsize / 2 + 1));
    d = min(boxsize / 2, max(d, -boxsize / 2 + 1));
    
    // Convert negative indices to positive (FFTW wrapping)
    if (r < 0) r = boxsize + r;
    if (d < 0) d = boxsize + d;
    
    // Final bounds check
    r = min(r, boxsize - 1);
    d = min(d, boxsize - 1);
    
    // Sample from texture (convert to 0.5-offset coordinates for texture sampling)
    float real_val = tex3D<float>(tex_real, c + 0.5f, r + 0.5f, d + 0.5f);
    float imag_val = tex3D<float>(tex_imag, c + 0.5f, r + 0.5f, d + 0.5f);
    cuFloatComplex value = make_cuFloatComplex(real_val, imag_val);
    
    // Return conjugated value if we used Friedel symmetry
    return need_conjugate ? complex_conj(value) : value;
}

// Sample from 3D FFTW-formatted Fourier space with automatic Friedel symmetry handling
__device__ __forceinline__ cuFloatComplex sample_3d_fftw_with_conjugate(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_depth_stride, int rec_row_stride,
    int d, int r, int c
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
    c = min(c, boxsize_half - 1);  // Column: [0, boxsize/2]
    
    // Row and depth: [-boxsize/2+1, boxsize/2]
    r = min(boxsize / 2, max(r, -boxsize / 2 + 1));
    d = min(boxsize / 2, max(d, -boxsize / 2 + 1));
    
    // Convert negative indices to positive (FFTW wrapping)
    if (r < 0) r = boxsize + r;
    if (d < 0) d = boxsize + d;
    
    // Final bounds check
    r = min(r, boxsize - 1);
    d = min(d, boxsize - 1);
    
    // Calculate linear index
    int idx = b * rec_batch_stride + d * rec_depth_stride + r * rec_row_stride + c;
    cuFloatComplex value = rec[idx];
    
    // Return conjugated value if we used Friedel symmetry
    return need_conjugate ? complex_conj(value) : value;
}

// Trilinear interpolation kernel with optional texture sampling
__device__ __forceinline__ cuFloatComplex trilinear_interpolate(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_depth_stride, int rec_row_stride,
    float d, float r, float c,
    cudaTextureObject_t tex_real = 0, cudaTextureObject_t tex_imag = 0, bool use_textures = false
) {
    // Extract integer and fractional parts of coordinates
    int d_floor = floorf(d);
    int r_floor = floorf(r);
    int c_floor = floorf(c);
    float d_frac = d - d_floor;  // Fractional part [0,1)
    float r_frac = r - r_floor;  // Fractional part [0,1)
    float c_frac = c - c_floor;  // Fractional part [0,1)

    // Sample 2x2x2 = 8 neighboring voxels
    // Using systematic naming: pDRC where D,R,C ∈ {0,1} indicate the offset
    cuFloatComplex p000, p001, p010, p011, p100, p101, p110, p111;
    
    if (use_textures) {
        p000 = sample_3d_texture(tex_real, tex_imag, boxsize, boxsize_half, d_floor,     r_floor,     c_floor);
        p001 = sample_3d_texture(tex_real, tex_imag, boxsize, boxsize_half, d_floor,     r_floor,     c_floor + 1);
        p010 = sample_3d_texture(tex_real, tex_imag, boxsize, boxsize_half, d_floor,     r_floor + 1, c_floor);
        p011 = sample_3d_texture(tex_real, tex_imag, boxsize, boxsize_half, d_floor,     r_floor + 1, c_floor + 1);
        p100 = sample_3d_texture(tex_real, tex_imag, boxsize, boxsize_half, d_floor + 1, r_floor,     c_floor);
        p101 = sample_3d_texture(tex_real, tex_imag, boxsize, boxsize_half, d_floor + 1, r_floor,     c_floor + 1);
        p110 = sample_3d_texture(tex_real, tex_imag, boxsize, boxsize_half, d_floor + 1, r_floor + 1, c_floor);
        p111 = sample_3d_texture(tex_real, tex_imag, boxsize, boxsize_half, d_floor + 1, r_floor + 1, c_floor + 1);
    } else {
        p000 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor,     c_floor);
        p001 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor,     c_floor + 1);
        p010 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor + 1, c_floor);
        p011 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor + 1, c_floor + 1);
        p100 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor,     c_floor);
        p101 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor,     c_floor + 1);
        p110 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor + 1, c_floor);
        p111 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor + 1, c_floor + 1);
    }

    // Trilinear interpolation: interpolate in each dimension sequentially
    // First, interpolate along the c dimension (4 edge interpolations)
    cuFloatComplex p00 = complex_add(p000, complex_scale(cuCsubf(p001, p000), c_frac));  // Back-bottom edge
    cuFloatComplex p01 = complex_add(p010, complex_scale(cuCsubf(p011, p010), c_frac));  // Back-top edge  
    cuFloatComplex p10 = complex_add(p100, complex_scale(cuCsubf(p101, p100), c_frac));  // Front-bottom edge
    cuFloatComplex p11 = complex_add(p110, complex_scale(cuCsubf(p111, p110), c_frac));  // Front-top edge

    // Second, interpolate along the r dimension (2 face interpolations)
    cuFloatComplex p0 = complex_add(p00, complex_scale(cuCsubf(p01, p00), r_frac));  // Back face
    cuFloatComplex p1 = complex_add(p10, complex_scale(cuCsubf(p11, p10), r_frac));  // Front face

    // Finally, interpolate along the d dimension (final result)
    return complex_add(p0, complex_scale(cuCsubf(p1, p0), d_frac));
}

// Tricubic interpolation kernel helper functions
__device__ __forceinline__ float tricubic_kernel(float s) {
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

// Safe 3D sampling with edge clamping for tricubic interpolation
__device__ __forceinline__ cuFloatComplex sample_3d_with_edge_clamping(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_depth_stride, int rec_row_stride,
    int d, int r, int c,
    cudaTextureObject_t tex_real = 0, cudaTextureObject_t tex_imag = 0, bool use_textures = false
) {
    // For c: after Friedel symmetry, clamp |c| to valid range [0, boxsize_half-1]
    if (abs(c) >= boxsize_half) {
        c = (c < 0) ? -(boxsize_half - 1) : (boxsize_half - 1);
    }
    
    // For r and d: clamp to valid range [-boxsize/2 + 1, boxsize/2]
    r = max(-boxsize / 2 + 1, min(r, boxsize / 2));
    d = max(-boxsize / 2 + 1, min(d, boxsize / 2));
    
    if (use_textures) {
        return sample_3d_texture(tex_real, tex_imag, boxsize, boxsize_half, d, r, c);
    } else {
        return sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d, r, c);
    }
}

// Tricubic interpolation kernel with optional texture sampling
__device__ __forceinline__ cuFloatComplex tricubic_interpolate(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_depth_stride, int rec_row_stride,
    float d, float r, float c,
    cudaTextureObject_t tex_real = 0, cudaTextureObject_t tex_imag = 0, bool use_textures = false
) {
    // Extract integer and fractional parts
    int d_floor = floorf(d);
    int r_floor = floorf(r);
    int c_floor = floorf(c);
    float d_frac = d - d_floor;
    float r_frac = r - r_floor;
    float c_frac = c - c_floor;
    
    cuFloatComplex result = make_cuFloatComplex(0.0f, 0.0f);
    
    // Sample 4x4x4 = 64 grid around the interpolation point
    // Grid extends from (d_floor-1, r_floor-1, c_floor-1) to (d_floor+2, r_floor+2, c_floor+2)
    for (int k = -1; k <= 2; ++k) {      // Depth offset: covers 4 depths
        float weight_d = tricubic_kernel(d_frac - k);
        
        for (int i = -1; i <= 2; ++i) {  // Row offset: covers 4 rows
            float weight_r = tricubic_kernel(r_frac - i);

            for (int j = -1; j <= 2; ++j) {  // Column offset: covers 4 columns
                cuFloatComplex sample = sample_3d_with_edge_clamping(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                                                    d_floor + k, r_floor + i, c_floor + j,
                                                                    tex_real, tex_imag, use_textures);
                // Compute tricubic weights for this grid position (separable)
                float weight_c = tricubic_kernel(c_frac - j);
                // Accumulate weighted contribution
                result = complex_add(result, complex_scale(sample, weight_d * weight_r * weight_c));
            }
        }
    }
    
    return result;
}

// Forward projection CUDA kernel for 3D->2D
__global__ void project_3d_to_2d_forw_kernel(
    const cuFloatComplex* reconstruction,
    const float* rotations,
    const float* shifts,
    cuFloatComplex* projections,
    CudaParams3D params
) {
    // Optimized thread organization: each block handles one entire projection
    // Grid: (P, B, 1), Block: (256, 1, 1)
    int p = blockIdx.x;  // Pose index
    int b = blockIdx.y;  // Batch index
    
    // Check bounds for batch and pose
    if (p >= params.P || b >= params.B) {
        return;
    }
    
    // Pre-compute shared parameters for this projection (pose p, batch b)
    
    // 3x3 Rotation matrix (shared for all pixels in this projection)
    int rot_b_idx = (params.B_rot == 1) ? 0 : b;
    int rot_idx_base = ((rot_b_idx * params.P + p) * 3 * 3);  // Index to start of 3x3 matrix
    float rot_00 = rotations[rot_idx_base + 0];     // R[0][0]
    float rot_01 = rotations[rot_idx_base + 1];     // R[0][1] 
    float rot_02 = rotations[rot_idx_base + 2];     // R[0][2]
    float rot_10 = rotations[rot_idx_base + 3];     // R[1][0]
    float rot_11 = rotations[rot_idx_base + 4];     // R[1][1]
    float rot_12 = rotations[rot_idx_base + 5];     // R[1][2]
    float rot_20 = rotations[rot_idx_base + 6];     // R[2][0]
    float rot_21 = rotations[rot_idx_base + 7];     // R[2][1]
    float rot_22 = rotations[rot_idx_base + 8];     // R[2][2]
    
    // Shifts (shared for all pixels in this projection)
    float shift_r = 0.0f, shift_c = 0.0f;
    if (params.has_shifts) {
        int shift_b_idx = (params.B_shift == 1) ? 0 : b;
        int shift_idx_base = ((shift_b_idx * params.P + p) * 2);
        shift_r = shifts[shift_idx_base + 0];
        shift_c = shifts[shift_idx_base + 1];
    }
    
    // Pre-compute constants for this projection
    int proj_batch_stride = params.P * params.proj_boxsize * params.proj_boxsize_half;
    int proj_pose_stride = params.proj_boxsize * params.proj_boxsize_half;
    int proj_row_stride = params.proj_boxsize_half;
    int proj_base_idx = b * proj_batch_stride + p * proj_pose_stride;
    
    // 4D reconstruction strides: [B, D, H, W/2+1]
    int rec_batch_stride = params.boxsize * params.boxsize * params.boxsize_half;  // B stride
    int rec_depth_stride = params.boxsize * params.boxsize_half;                   // D stride  
    int rec_row_stride = params.boxsize_half;                                      // H stride
    
    float fourier_cutoff_sq = params.fourier_radius_cutoff * params.fourier_radius_cutoff;
    
    // Loop over all pixels in this projection, with threads cooperating
    int total_pixels = params.proj_boxsize * params.proj_boxsize_half;
    
    for (int pixel_idx = threadIdx.x; pixel_idx < total_pixels; pixel_idx += blockDim.x) {
        // Convert linear pixel index to (i, j) coordinates
        int i = pixel_idx / params.proj_boxsize_half;  // Row
        int j = pixel_idx % params.proj_boxsize_half;  // Column
        
        // Convert array indices to 2D Fourier coordinates (must match CPU logic exactly)
        float proj_coord_c = float(j);  // Column: always positive (FFTW half-space)
        float proj_coord_r = (i <= params.proj_boxsize / 2) ? float(i) : float(i) - float(params.proj_boxsize); // Row: handle wrap-around
        
        // Apply Fourier space filtering (low-pass)
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            projections[proj_base_idx + pixel_idx] = make_cuFloatComplex(0.0f, 0.0f);
            continue;
        }
        
        // Apply oversampling scaling to 2D coordinates
        // Oversampling > 1 simulates zero-padding in real space
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Central slice: extend 2D coordinates to 3D with d=0
        float sample_d = 0.0f;  // Central slice through origin
        
        // Apply 3x3 rotation matrix to get sampling coordinates in 3D reconstruction
        // Matrix multiplication: [rot_c; rot_r; rot_d] = R * [sample_c; sample_r; sample_d]
        float rot_c = rot_00 * sample_c + rot_01 * sample_r + rot_02 * sample_d;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r + rot_12 * sample_d;
        float rot_d = rot_20 * sample_c + rot_21 * sample_r + rot_22 * sample_d;
        
        // Interpolate from 4D reconstruction at rotated 3D coordinates
        cuFloatComplex val;
        if (params.interpolation_method == 0) {  // linear (trilinear)
            val = trilinear_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                      rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c,
                                      params.tex_real, params.tex_imag, params.use_textures);
        } else {  // cubic (tricubic)
            val = tricubic_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                     rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c,
                                     params.tex_real, params.tex_imag, params.use_textures);
        }
        
        // Apply phase shift if translations are provided (using pre-computed shifts)
        // Shift in real space = phase modulation in Fourier space
        if (params.has_shifts) {
            float phase = -2.0f * M_PI * (proj_coord_r * shift_r / params.proj_boxsize + 
                                          proj_coord_c * shift_c / params.proj_boxsize);
            cuFloatComplex phase_factor = make_cuFloatComplex(cosf(phase), sinf(phase));
            val = complex_mul(val, phase_factor);
        }
        
        projections[proj_base_idx + pixel_idx] = val;
    }
}

// Atomic operations for backward projection gradient accumulation
__device__ __forceinline__ void atomic_add_complex(cuFloatComplex* target, cuFloatComplex value) {
    // CUDA doesn't have atomic operations for complex numbers directly
    // We need to atomically add real and imaginary parts separately
    float* real_ptr = reinterpret_cast<float*>(target);
    float* imag_ptr = real_ptr + 1;
    
    atomicAdd(real_ptr, cuCrealf(value));
    atomicAdd(imag_ptr, cuCimagf(value)); 
}

__device__ __forceinline__ void atomic_add_real(float* target, float value) {
    atomicAdd(target, value);
}

// Trilinear interpolation with gradients for backward pass
__device__ __forceinline__ void trilinear_interpolate_with_gradients(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_depth_stride, int rec_row_stride,
    float d, float r, float c,
    cuFloatComplex* val, cuFloatComplex* grad_d, cuFloatComplex* grad_r, cuFloatComplex* grad_c
) {
    int d_floor = floorf(d);
    int r_floor = floorf(r);
    int c_floor = floorf(c);
    
    float d_frac = d - d_floor;
    float r_frac = r - r_floor;
    float c_frac = c - c_floor;

    // Sample 2x2x2 grid
    cuFloatComplex p000 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor,     c_floor);
    cuFloatComplex p001 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor,     c_floor + 1);
    cuFloatComplex p010 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor + 1, c_floor);
    cuFloatComplex p011 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor + 1, c_floor + 1);
    cuFloatComplex p100 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor,     c_floor);
    cuFloatComplex p101 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor,     c_floor + 1);
    cuFloatComplex p110 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor + 1, c_floor);
    cuFloatComplex p111 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor + 1, c_floor + 1);

    // Value computation (same as interpolate method)
    cuFloatComplex p00 = complex_add(p000, complex_scale(cuCsubf(p001, p000), c_frac));
    cuFloatComplex p01 = complex_add(p010, complex_scale(cuCsubf(p011, p010), c_frac));
    cuFloatComplex p10 = complex_add(p100, complex_scale(cuCsubf(p101, p100), c_frac));
    cuFloatComplex p11 = complex_add(p110, complex_scale(cuCsubf(p111, p110), c_frac));
    
    cuFloatComplex p0 = complex_add(p00, complex_scale(cuCsubf(p01, p00), r_frac));
    cuFloatComplex p1 = complex_add(p10, complex_scale(cuCsubf(p11, p10), r_frac));
    
    *val = complex_add(p0, complex_scale(cuCsubf(p1, p0), d_frac));

    // Analytical spatial gradients derived from trilinear formula
    // ∂f/∂d, ∂f/∂r, ∂f/∂c computed analytically for efficiency
    *grad_d = cuCsubf(p1, p0);  // ∂f/∂d
    *grad_r = complex_add(complex_scale(cuCsubf(p01, p00), 1.0f - d_frac), complex_scale(cuCsubf(p11, p10), d_frac));  // ∂f/∂r
    *grad_c = complex_add(complex_add(complex_scale(cuCsubf(p001, p000), (1.0f - d_frac) * (1.0f - r_frac)),
                                     complex_scale(cuCsubf(p011, p010), (1.0f - d_frac) * r_frac)),
                         complex_add(complex_scale(cuCsubf(p101, p100), d_frac * (1.0f - r_frac)),
                                    complex_scale(cuCsubf(p111, p110), d_frac * r_frac)));  // ∂f/∂c
}

// Tricubic kernel derivative for gradient computation
__device__ __forceinline__ float tricubic_kernel_derivative(float s) {
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

// Tricubic interpolation with gradients for backward pass
__device__ __forceinline__ void tricubic_interpolate_with_gradients(
    const cuFloatComplex* rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_depth_stride, int rec_row_stride,
    float d, float r, float c,
    cuFloatComplex* val, cuFloatComplex* grad_d, cuFloatComplex* grad_r, cuFloatComplex* grad_c
) {
    int d_floor = floorf(d);
    int r_floor = floorf(r);
    int c_floor = floorf(c);
    
    float d_frac = d - d_floor;
    float r_frac = r - r_floor;
    float c_frac = c - c_floor;
    
    *val = make_cuFloatComplex(0.0f, 0.0f);       // Interpolated value
    *grad_d = make_cuFloatComplex(0.0f, 0.0f);    // Gradient w.r.t. depth coordinate
    *grad_r = make_cuFloatComplex(0.0f, 0.0f);    // Gradient w.r.t. row coordinate
    *grad_c = make_cuFloatComplex(0.0f, 0.0f);    // Gradient w.r.t. column coordinate
    
    // Sample 4x4x4 grid and compute value + gradients simultaneously
    for (int k = -1; k <= 2; ++k) {
        float weight_d = tricubic_kernel(d_frac - k);
        float dweight_d = tricubic_kernel_derivative(d_frac - k);
        
        for (int i = -1; i <= 2; ++i) {
            float weight_r = tricubic_kernel(r_frac - i);
            float dweight_r = tricubic_kernel_derivative(r_frac - i);
            
            for (int j = -1; j <= 2; ++j) {
                cuFloatComplex sample = sample_3d_with_edge_clamping(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                                                    d_floor + k, r_floor + i, c_floor + j);
                
                // Compute weights and their derivatives for this grid position
                float weight_c = tricubic_kernel(c_frac - j);
                float dweight_c = tricubic_kernel_derivative(c_frac - j);
                
                // Accumulate value and gradients (separable cubic kernels)
                *val = complex_add(*val, complex_scale(sample, weight_d * weight_r * weight_c));
                *grad_d = complex_add(*grad_d, complex_scale(sample, dweight_d * weight_r * weight_c));  // Chain rule: ∂f/∂d
                *grad_r = complex_add(*grad_r, complex_scale(sample, weight_d * dweight_r * weight_c));  // Chain rule: ∂f/∂r
                *grad_c = complex_add(*grad_c, complex_scale(sample, weight_d * weight_r * dweight_c));  // Chain rule: ∂f/∂c
            }
        }
    }
}

// Helper function to distribute trilinear gradient to 3D reconstruction
__device__ __forceinline__ void distribute_trilinear_gradient(
    cuFloatComplex* grad_rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_depth_stride, int rec_row_stride,
    float d, float r, float c, cuFloatComplex grad_val
) {
    // Extract integer and fractional parts
    int d_floor = floorf(d);
    int r_floor = floorf(r);
    int c_floor = floorf(c);
    float d_frac = d - d_floor;
    float r_frac = r - r_floor;
    float c_frac = c - c_floor;

    // Distribute gradient to 2x2x2 neighborhood with trilinear weights
    // Each contribution needs 3D Friedel symmetry handling
    auto accumulate_grad = [&](int grid_d, int grid_r, int grid_c, cuFloatComplex weight_grad) {
        bool needs_conj = false;
        
        // Handle 3D Friedel symmetry for negative column indices
        if (grid_c < 0) { 
            grid_c = -grid_c;           // Mirror column to positive side
            grid_r = -grid_r;           // Mirror row as well
            grid_d = -grid_d;           // Mirror depth as well
            needs_conj = true;          // Need to conjugate the value
        }
        
        // Bounds checking
        if (grid_c >= boxsize_half) return;  // Beyond stored frequency range
        if (grid_r > boxsize / 2 || grid_r < -boxsize / 2 + 1) return;  // Beyond valid row range
        if (grid_d > boxsize / 2 || grid_d < -boxsize / 2 + 1) return;  // Beyond valid depth range

        // Convert negative row indices to positive (FFTW wrapping)
        int r_eff = grid_r < 0 ? boxsize + grid_r : grid_r;
        int d_eff = grid_d < 0 ? boxsize + grid_d : grid_d;
        if (r_eff >= boxsize || d_eff >= boxsize) return;  // Final bounds check

        // Calculate index and atomically accumulate gradient
        int idx = b * rec_batch_stride + d_eff * rec_depth_stride + r_eff * rec_row_stride + grid_c;
        cuFloatComplex final_grad = needs_conj ? complex_conj(weight_grad) : weight_grad;
        atomic_add_complex(&grad_rec[idx], final_grad);
    };

    // Distribute to 2x2x2 neighborhood
    accumulate_grad(d_floor,     r_floor,     c_floor,     complex_scale(grad_val, (1.0f - d_frac) * (1.0f - r_frac) * (1.0f - c_frac))); // p000
    accumulate_grad(d_floor,     r_floor,     c_floor + 1, complex_scale(grad_val, (1.0f - d_frac) * (1.0f - r_frac) * c_frac));       // p001
    accumulate_grad(d_floor,     r_floor + 1, c_floor,     complex_scale(grad_val, (1.0f - d_frac) * r_frac * (1.0f - c_frac)));       // p010
    accumulate_grad(d_floor,     r_floor + 1, c_floor + 1, complex_scale(grad_val, (1.0f - d_frac) * r_frac * c_frac));             // p011
    accumulate_grad(d_floor + 1, r_floor,     c_floor,     complex_scale(grad_val, d_frac * (1.0f - r_frac) * (1.0f - c_frac)));       // p100
    accumulate_grad(d_floor + 1, r_floor,     c_floor + 1, complex_scale(grad_val, d_frac * (1.0f - r_frac) * c_frac));             // p101
    accumulate_grad(d_floor + 1, r_floor + 1, c_floor,     complex_scale(grad_val, d_frac * r_frac * (1.0f - c_frac)));             // p110
    accumulate_grad(d_floor + 1, r_floor + 1, c_floor + 1, complex_scale(grad_val, d_frac * r_frac * c_frac));                   // p111
}

// Helper function to distribute tricubic gradient to 3D reconstruction
__device__ __forceinline__ void distribute_tricubic_gradient(
    cuFloatComplex* grad_rec,
    int b, int boxsize, int boxsize_half,
    int rec_batch_stride, int rec_depth_stride, int rec_row_stride,
    float d, float r, float c, cuFloatComplex grad_val
) {
    // Extract integer and fractional parts
    int d_floor = floorf(d);
    int r_floor = floorf(r);
    int c_floor = floorf(c);
    float d_frac = d - d_floor;
    float r_frac = r - r_floor;
    float c_frac = c - c_floor;
    
    // Same 3D Friedel symmetry handling function
    auto accumulate_grad = [&](int grid_d, int grid_r, int grid_c, cuFloatComplex weight_grad) {
        bool needs_conj = false;
        
        if (grid_c < 0) { 
            grid_c = -grid_c;
            grid_r = -grid_r;
            grid_d = -grid_d;
            needs_conj = true;
        }
        
        if (grid_c >= boxsize_half) return;
        if (grid_r > boxsize / 2 || grid_r < -boxsize / 2 + 1) return;
        if (grid_d > boxsize / 2 || grid_d < -boxsize / 2 + 1) return;

        int r_eff = grid_r < 0 ? boxsize + grid_r : grid_r;
        int d_eff = grid_d < 0 ? boxsize + grid_d : grid_d;
        if (r_eff >= boxsize || d_eff >= boxsize) return;

        int idx = b * rec_batch_stride + d_eff * rec_depth_stride + r_eff * rec_row_stride + grid_c;
        cuFloatComplex final_grad = needs_conj ? complex_conj(weight_grad) : weight_grad;
        atomic_add_complex(&grad_rec[idx], final_grad);
    };
    
    // Distribute gradient to 4x4x4 neighborhood using tricubic weights
    for (int k = -1; k <= 2; ++k) {      // Depth offset: covers 4 depths
        float weight_d = tricubic_kernel(d_frac - k);

        for (int i = -1; i <= 2; ++i) {  // Row offset: covers 4 rows
            float weight_r = tricubic_kernel(r_frac - i);

            for (int j = -1; j <= 2; ++j) {  // Column offset: covers 4 columns
                float weight_c = tricubic_kernel(c_frac - j);
                float total_weight = weight_d * weight_r * weight_c;
                
                // Only distribute if weight is non-zero (tricubic has finite support)
                if (total_weight != 0.0f) {
                    accumulate_grad(d_floor + k, r_floor + i, c_floor + j, complex_scale(grad_val, total_weight));
                }
            }
        }
    }
}

// Backward projection CUDA kernel for 3D->2D
__global__ void project_3d_to_2d_back_kernel(
    const cuFloatComplex* grad_projections,
    const cuFloatComplex* reconstruction,
    const float* rotations,
    const float* shifts,
    cuFloatComplex* grad_reconstruction,
    float* grad_rotations,
    float* grad_shifts,
    CudaParams3D params
) {
    // Thread organization: each block handles one projection (pose p, batch b)
    // Grid: (P, B, 1), Block: (256, 1, 1)
    int p = blockIdx.x;  // Pose index
    int b = blockIdx.y;  // Batch index
    
    // Check bounds
    if (p >= params.P || b >= params.B) {
        return;
    }
    
    // Extract gradient flags from interpolation_method upper bits
    bool need_rotation_grads = (params.interpolation_method & 0x10) != 0;
    bool need_shift_grads = (params.interpolation_method & 0x20) != 0;
    int interpolation_method = params.interpolation_method & 0x0F;  // Lower 4 bits
    
    // Pre-compute parameters for this projection
    int rot_b_idx = (params.B_rot == 1) ? 0 : b;
    int rot_idx_base = ((rot_b_idx * params.P + p) * 3 * 3);  // 3x3 matrix
    float rot_00 = rotations[rot_idx_base + 0];
    float rot_01 = rotations[rot_idx_base + 1]; 
    float rot_02 = rotations[rot_idx_base + 2];
    float rot_10 = rotations[rot_idx_base + 3];
    float rot_11 = rotations[rot_idx_base + 4];
    float rot_12 = rotations[rot_idx_base + 5];
    float rot_20 = rotations[rot_idx_base + 6];
    float rot_21 = rotations[rot_idx_base + 7];
    float rot_22 = rotations[rot_idx_base + 8];
    
    float shift_r = 0.0f, shift_c = 0.0f;
    if (params.has_shifts) {
        int shift_b_idx = (params.B_shift == 1) ? 0 : b;
        int shift_idx_base = ((shift_b_idx * params.P + p) * 2);
        shift_r = shifts[shift_idx_base + 0];
        shift_c = shifts[shift_idx_base + 1];
    }
    
    // Pre-compute strides
    int proj_batch_stride = params.P * params.proj_boxsize * params.proj_boxsize_half;
    int proj_pose_stride = params.proj_boxsize * params.proj_boxsize_half;
    int proj_base_idx = b * proj_batch_stride + p * proj_pose_stride;
    
    // 4D reconstruction strides: [B, D, H, W/2+1]
    int rec_batch_stride = params.boxsize * params.boxsize * params.boxsize_half;
    int rec_depth_stride = params.boxsize * params.boxsize_half;
    int rec_row_stride = params.boxsize_half;
    
    float fourier_cutoff_sq = params.fourier_radius_cutoff * params.fourier_radius_cutoff;
    
    // Local accumulators for rotation and shift gradients
    float local_rot_grad[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // 3x3 matrix flattened
    float local_shift_grad[2] = {0.0f, 0.0f};
    
    // Process all pixels in this projection
    int total_pixels = params.proj_boxsize * params.proj_boxsize_half;
    
    for (int pixel_idx = threadIdx.x; pixel_idx < total_pixels; pixel_idx += blockDim.x) {
        int i = pixel_idx / params.proj_boxsize_half;
        int j = pixel_idx % params.proj_boxsize_half;

        if (j == 0 && i >= params.proj_boxsize / 2) {
            // Skip Friedel-symmetric half of the x = 0 line (handled by other half)
            continue;
        }
        
        // Convert to Fourier coordinates
        float proj_coord_c = float(j);
        float proj_coord_r = (i <= params.proj_boxsize / 2) ? float(i) : float(i) - float(params.proj_boxsize);
        
        // Apply frequency cutoff
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            continue;
        }
        
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Central slice: extend 2D coordinates to 3D with d=0
        float sample_d = 0.0f;  // Central slice
        
        float rot_c = rot_00 * sample_c + rot_01 * sample_r + rot_02 * sample_d;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r + rot_12 * sample_d;
        float rot_d = rot_20 * sample_c + rot_21 * sample_r + rot_22 * sample_d;
        
        // Get gradient from projection
        cuFloatComplex grad_proj = grad_projections[proj_base_idx + pixel_idx];
        
        // Apply phase shift correction to gradient if shifts are present
        if (params.has_shifts) {
            float phase = 2.0f * M_PI * (proj_coord_r * shift_r / params.proj_boxsize + 
                                         proj_coord_c * shift_c / params.proj_boxsize);
            cuFloatComplex phase_factor = make_cuFloatComplex(cosf(phase), sinf(phase));
            grad_proj = complex_mul(grad_proj, phase_factor);
        }
        
        // Distribute gradient to reconstruction using appropriate interpolation method
        if (interpolation_method == 0) {  // linear
            distribute_trilinear_gradient(grad_reconstruction, b, params.boxsize, params.boxsize_half,
                                        rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c, grad_proj);
        } else {  // cubic
            distribute_tricubic_gradient(grad_reconstruction, b, params.boxsize, params.boxsize_half,
                                       rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c, grad_proj);
        }
        
        // Compute gradients w.r.t. shift parameters if needed
        if (need_shift_grads) {
            // Get reconstruction value for shift gradient computation
            cuFloatComplex rec_val;
            if (interpolation_method == 0) {
                rec_val = trilinear_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                              rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c);
            } else {
                rec_val = tricubic_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                             rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c);
            }
            
            // Apply phase modulation to reconstruction value
            if (params.has_shifts) {
                float phase = -2.0f * M_PI * (proj_coord_r * shift_r / params.proj_boxsize + 
                                              proj_coord_c * shift_c / params.proj_boxsize);
                cuFloatComplex phase_factor = make_cuFloatComplex(cosf(phase), sinf(phase));
                rec_val = complex_mul(rec_val, phase_factor);
            }
            
            // Compute phase derivatives
            cuFloatComplex phase_grad_r = make_cuFloatComplex(0.0f, -2.0f * M_PI * proj_coord_r / params.proj_boxsize);
            cuFloatComplex phase_grad_c = make_cuFloatComplex(0.0f, -2.0f * M_PI * proj_coord_c / params.proj_boxsize);
            phase_grad_r = complex_mul(phase_grad_r, rec_val);
            phase_grad_c = complex_mul(phase_grad_c, rec_val);
            
            // Accumulate shift gradients (real part of complex gradient)
            cuFloatComplex original_grad_proj = grad_projections[proj_base_idx + pixel_idx];
            local_shift_grad[0] += cuCrealf(complex_mul(original_grad_proj, complex_conj(phase_grad_r)));
            local_shift_grad[1] += cuCrealf(complex_mul(original_grad_proj, complex_conj(phase_grad_c)));
        }
        
        // Compute gradients w.r.t. rotation matrix if needed
        if (need_rotation_grads) {
            cuFloatComplex rec_val, grad_d, grad_r, grad_c;
            
            if (interpolation_method == 0) {
                trilinear_interpolate_with_gradients(reconstruction, b, params.boxsize, params.boxsize_half,
                                                   rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c,
                                                   &rec_val, &grad_d, &grad_r, &grad_c);
            } else {
                tricubic_interpolate_with_gradients(reconstruction, b, params.boxsize, params.boxsize_half,
                                                  rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c,
                                                  &rec_val, &grad_d, &grad_r, &grad_c);
            }
            
            // Chain rule: ∂f/∂R[i][j] = (∂f/∂rot_coord) * (∂rot_coord/∂R[i][j])
            // 
            // 3D Rotation transformation: [rot_c; rot_r; rot_d] = R * [sample_c; sample_r; sample_d]
            // Therefore:
            // ∂rot_c/∂R[0][0] = sample_c, ∂rot_c/∂R[0][1] = sample_r, ∂rot_c/∂R[0][2] = sample_d
            // ∂rot_r/∂R[1][0] = sample_c, ∂rot_r/∂R[1][1] = sample_r, ∂rot_r/∂R[1][2] = sample_d
            // ∂rot_d/∂R[2][0] = sample_c, ∂rot_d/∂R[2][1] = sample_r, ∂rot_d/∂R[2][2] = sample_d
            //
            // Accumulate gradients locally (taking real part for real-valued rotation matrices):
            local_rot_grad[0] += cuCrealf(complex_mul(grad_proj, complex_conj(complex_scale(grad_c, sample_c))));  // ∂f/∂R[0][0]
            local_rot_grad[1] += cuCrealf(complex_mul(grad_proj, complex_conj(complex_scale(grad_c, sample_r))));  // ∂f/∂R[0][1]
            local_rot_grad[2] += cuCrealf(complex_mul(grad_proj, complex_conj(complex_scale(grad_c, sample_d))));  // ∂f/∂R[0][2]
            local_rot_grad[3] += cuCrealf(complex_mul(grad_proj, complex_conj(complex_scale(grad_r, sample_c))));  // ∂f/∂R[1][0]
            local_rot_grad[4] += cuCrealf(complex_mul(grad_proj, complex_conj(complex_scale(grad_r, sample_r))));  // ∂f/∂R[1][1]
            local_rot_grad[5] += cuCrealf(complex_mul(grad_proj, complex_conj(complex_scale(grad_r, sample_d))));  // ∂f/∂R[1][2]
            local_rot_grad[6] += cuCrealf(complex_mul(grad_proj, complex_conj(complex_scale(grad_d, sample_c))));  // ∂f/∂R[2][0]
            local_rot_grad[7] += cuCrealf(complex_mul(grad_proj, complex_conj(complex_scale(grad_d, sample_r))));  // ∂f/∂R[2][1]
            local_rot_grad[8] += cuCrealf(complex_mul(grad_proj, complex_conj(complex_scale(grad_d, sample_d))));  // ∂f/∂R[2][2]
        }
    }
    
    // Atomically add local gradients to global tensors
    if (need_rotation_grads) {
        for (int i = 0; i < 9; ++i) {
            atomic_add_real(&grad_rotations[rot_idx_base + i], local_rot_grad[i]);
        }
    }
    
    if (need_shift_grads) {
        int shift_b_idx = (params.B_shift == 1) ? 0 : b;
        int shift_idx_base = ((shift_b_idx * params.P + p) * 2);
        atomic_add_real(&grad_shifts[shift_idx_base + 0], local_shift_grad[0]);
        atomic_add_real(&grad_shifts[shift_idx_base + 1], local_shift_grad[1]);
    }
}

// Forward projection from 4D Fourier reconstruction to 2D projections (CUDA version)
at::Tensor project_3d_to_2d_forw_cuda(
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef output_shape,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Input validation
    TORCH_CHECK(reconstruction.is_cuda(), "Input reconstruction must be on CUDA device");
    TORCH_CHECK(rotations.is_cuda(), "Input rotations must be on CUDA device");
    TORCH_CHECK(reconstruction.is_complex(), "Reconstruction must be a complex tensor");
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic",
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(reconstruction.dim() == 4,
                "Reconstruction must be a 4D tensor (B, D, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 3 && rotations.size(3) == 3,
                "Rotations must be (B_rot, P, 3, 3)");

    if (shifts.has_value()) {
        TORCH_CHECK(shifts->is_cuda(), "Shifts must be on CUDA device");
        TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
        TORCH_CHECK(shifts->size(0) == reconstruction.size(0) || shifts->size(0) == 1,
                    "Batch size of shifts must be 1 or same as reconstruction");
        TORCH_CHECK(shifts->size(1) == rotations.size(1),
                    "Number of poses in shifts must match rotations");
        TORCH_CHECK(shifts->scalar_type() == rotations.scalar_type(),
                    "Shifts and rotations must have the same dtype");
    }

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

    // Set CUDA device guard to ensure operations happen on the right device
    const c10::cuda::CUDAGuard device_guard(reconstruction.device());
    
    // Get CUDA stream for asynchronous operations
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Validate shifts tensor if provided
    c10::optional<at::Tensor> shifts_contiguous;
    int64_t B_shift = 1;
    if (shifts.has_value()) {
        TORCH_CHECK(shifts->is_cuda(), "Shifts must be on CUDA device");
        TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
        TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1, "Batch size of shifts must be 1 or same as reconstruction");
        TORCH_CHECK(shifts->size(1) == P, "Number of poses in shifts must match rotations");
        TORCH_CHECK(shifts->scalar_type() == rotations.scalar_type(), "Shifts and rotations must have the same dtype");
        B_shift = shifts->size(0);
        shifts_contiguous = shifts->is_contiguous() ? *shifts : shifts->contiguous();
    }

    // Ensure tensors are contiguous
    auto rec_contiguous = reconstruction.is_contiguous() ? reconstruction : reconstruction.contiguous();
    auto rot_contiguous = rotations.is_contiguous() ? rotations : rotations.contiguous();
    auto proj_contiguous = projection.is_contiguous() ? projection : projection.contiguous();

    // Set up kernel parameters
    CudaParams3D params = {
        (int)B, (int)P, (int)boxsize, (int)boxsize_half,
        (int)proj_boxsize, (int)proj_boxsize_half, (int)B_rot, (int)B_shift,
        (int)(shifts.has_value() ? 1 : 0),
        (interpolation == "linear") ? 0 : 1,
        static_cast<float>(oversampling),
        static_cast<float>(fourier_radius_cutoff.value_or(proj_boxsize / 2.0)),
        0, 0, false  // tex_real, tex_imag, use_textures - will be set below
    };

    // Launch CUDA kernel
    // Grid: (P, B, 1), Block: (256, 1, 1) - each block handles one projection
    dim3 gridDim(P, B, 1);
    dim3 blockDim(256, 1, 1);

    // Check if we need double precision - fall back to CPU for gradcheck
    if (reconstruction.scalar_type() == at::kComplexDouble) {
        // For double precision (used by gradcheck), fall back to CPU since we don't 
        // want to implement full double precision CUDA kernels just for testing
        auto reconstruction_cpu = reconstruction.cpu();
        auto rotations_cpu = rotations.cpu();
        c10::optional<at::Tensor> shifts_cpu;
        if (shifts.has_value()) {
            shifts_cpu = shifts->cpu();
        }
        
        auto result_cpu = project_3d_to_2d_forw_cpu(
            reconstruction_cpu, rotations_cpu, shifts_cpu,
            output_shape, interpolation, oversampling, fourier_radius_cutoff
        );
        
        return result_cpu.to(reconstruction.device(), /*non_blocking=*/false);
    }

    // Create texture objects for B=1 optimization
    cudaTextureObject_t tex_real = 0, tex_imag = 0;
    cudaArray_t real_array = nullptr, imag_array = nullptr;
    at::Tensor real_part, imag_part;
    
    if (B == 1) {
        // Extract real and imaginary parts
        real_part = torch::real(reconstruction).squeeze(0);  // Remove batch dimension: [D, H, W/2+1]
        imag_part = torch::imag(reconstruction).squeeze(0);
        
        // Ensure contiguous memory layout
        real_part = real_part.contiguous();
        imag_part = imag_part.contiguous();
        
        // Create texture objects
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        
        // Configure texture descriptor
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.filterMode = cudaFilterModePoint;  // No hardware interpolation as requested
        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.addressMode[1] = cudaAddressModeBorder;
        texDesc.addressMode[2] = cudaAddressModeBorder;
        
        // Create CUDA arrays for 3D textures
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaExtent extent = make_cudaExtent(boxsize_half, boxsize, boxsize);
        
        // Real part array and texture
        cudaMalloc3DArray(&real_array, &channelDesc, extent);
        
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr(real_part.data_ptr<float>(), 
                                              boxsize_half * sizeof(float), 
                                              boxsize_half, boxsize);
        copyParams.dstArray = real_array;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&copyParams);
        
        resDesc.res.array.array = real_array;
        cudaCreateTextureObject(&tex_real, &resDesc, &texDesc, nullptr);
        
        // Imaginary part array and texture
        cudaMalloc3DArray(&imag_array, &channelDesc, extent);
        
        copyParams.srcPtr = make_cudaPitchedPtr(imag_part.data_ptr<float>(), 
                                              boxsize_half * sizeof(float), 
                                              boxsize_half, boxsize);
        copyParams.dstArray = imag_array;
        cudaMemcpy3D(&copyParams);
        
        resDesc.res.array.array = imag_array;
        cudaCreateTextureObject(&tex_imag, &resDesc, &texDesc, nullptr);
        
        // Update parameters to use textures
        params.tex_real = tex_real;
        params.tex_imag = tex_imag;
        params.use_textures = true;
    }

    // Get raw pointers for kernel launch (float32 only)
    const cuFloatComplex* rec_ptr = reinterpret_cast<const cuFloatComplex*>(rec_contiguous.data_ptr<c10::complex<float>>());
    const float* rot_ptr = rot_contiguous.data_ptr<float>();
    const float* shift_ptr = shifts.has_value() ? shifts_contiguous->data_ptr<float>() : nullptr;
    cuFloatComplex* proj_ptr = reinterpret_cast<cuFloatComplex*>(proj_contiguous.data_ptr<c10::complex<float>>());

    project_3d_to_2d_forw_kernel<<<gridDim, blockDim, 0, stream>>>(
        rec_ptr, rot_ptr, shift_ptr, proj_ptr, params
    );

    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Cleanup texture objects if they were created
    if (B == 1 && params.use_textures) {
        cudaDestroyTextureObject(tex_real);
        cudaDestroyTextureObject(tex_imag);
        
        // Free the CUDA arrays
        if (real_array) cudaFreeArray(real_array);
        if (imag_array) cudaFreeArray(imag_array);
    }

    // Copy back if projection wasn't originally contiguous
    if (!projection.is_contiguous()) {
        projection.copy_(proj_contiguous);
        return projection;
    }
    
    return proj_contiguous;
}

// Backward projection for gradients (CUDA version) 
std::tuple<at::Tensor, at::Tensor, at::Tensor> project_3d_to_2d_back_cuda(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Input validation
    TORCH_CHECK(grad_projections.is_cuda(), "Input grad_projections must be on CUDA device");
    TORCH_CHECK(reconstruction.is_cuda(), "Input reconstruction must be on CUDA device");
    TORCH_CHECK(rotations.is_cuda(), "Input rotations must be on CUDA device");
    TORCH_CHECK(grad_projections.is_complex(), "Grad projections must be a complex tensor");
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic",
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(grad_projections.dim() == 4,
                "Grad projections must be a 4D tensor (B, P, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 3 && rotations.size(3) == 3,
                "Rotations must be (B_rot, P, 3, 3)");

    if (shifts.has_value()) {
        TORCH_CHECK(shifts->is_cuda(), "Shifts must be on CUDA device");
        TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
        TORCH_CHECK(shifts->size(0) == reconstruction.size(0) || shifts->size(0) == 1,
                    "Batch size of shifts must be 1 or same as reconstruction");
        TORCH_CHECK(shifts->size(1) == rotations.size(1),
                    "Number of poses in shifts must match rotations");
        TORCH_CHECK(shifts->scalar_type() == rotations.scalar_type(),
                    "Shifts and rotations must have the same dtype");
    }

    const auto B = grad_projections.size(0);
    const auto P = grad_projections.size(1);
    const auto proj_boxsize = grad_projections.size(2);
    const auto proj_boxsize_half = grad_projections.size(3);
    
    const auto rec_depth = reconstruction.size(1);
    const auto rec_boxsize = reconstruction.size(2);
    const auto rec_boxsize_half = reconstruction.size(3);
    
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

    // Set CUDA device guard
    const c10::cuda::CUDAGuard device_guard(grad_projections.device());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Validate and prepare shifts
    c10::optional<at::Tensor> shifts_contiguous;
    const auto B_rot = rotations.size(0);
    int64_t B_shift = 1;
    if (shifts.has_value()) {
        TORCH_CHECK(shifts->is_cuda(), "Shifts must be on CUDA device");
        TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
        TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1, "Batch size of shifts must be 1 or same as reconstruction");
        TORCH_CHECK(shifts->size(1) == P, "Number of poses in shifts must match rotations");
        TORCH_CHECK(shifts->scalar_type() == rotations.scalar_type(), "Shifts and rotations must have the same dtype");
        B_shift = shifts->size(0);
        shifts_contiguous = shifts->is_contiguous() ? *shifts : shifts->contiguous();
    }

    // Ensure tensors are contiguous
    auto grad_proj_contiguous = grad_projections.is_contiguous() ? grad_projections : grad_projections.contiguous();
    auto rec_contiguous = reconstruction.is_contiguous() ? reconstruction : reconstruction.contiguous();
    auto rot_contiguous = rotations.is_contiguous() ? rotations : rotations.contiguous();
    auto grad_rec_contiguous = grad_reconstruction.is_contiguous() ? grad_reconstruction : grad_reconstruction.contiguous();
    auto grad_rot_contiguous = need_rotation_grads ? (grad_rotations.is_contiguous() ? grad_rotations : grad_rotations.contiguous()) : grad_rotations;
    auto grad_shifts_contiguous = need_shift_grads ? (grad_shifts.is_contiguous() ? grad_shifts : grad_shifts.contiguous()) : grad_shifts;

    // Set up kernel parameters with gradient flags
    CudaParams3D params = {
        (int)B, (int)P, (int)rec_boxsize, (int)rec_boxsize_half,
        (int)proj_boxsize, (int)proj_boxsize_half, (int)B_rot, (int)B_shift,
        (int)(shifts.has_value() ? 1 : 0),
        (interpolation == "linear" ? 0 : 1) | 
        (need_rotation_grads ? 0x10 : 0) | 
        (need_shift_grads ? 0x20 : 0),  // Pack flags into upper bits
        static_cast<float>(oversampling),
        static_cast<float>(fourier_radius_cutoff.value_or(proj_boxsize / 2.0)),
        0, 0, false  // tex_real, tex_imag, use_textures - not used in backward pass
    };

    // Launch CUDA kernel
    dim3 gridDim(P, B, 1);
    dim3 blockDim(256, 1, 1);

    // Check if we need double precision - fall back to CPU for gradcheck
    if (grad_projections.scalar_type() == at::kComplexDouble) {
        // For double precision (used by gradcheck), fall back to CPU
        auto grad_projections_cpu = grad_projections.cpu();
        auto reconstruction_cpu = reconstruction.cpu();
        auto rotations_cpu = rotations.cpu();
        c10::optional<at::Tensor> shifts_cpu;
        if (shifts.has_value()) {
            shifts_cpu = shifts->cpu();
        }
        
        auto [grad_reconstruction_cpu, grad_rotations_cpu, grad_shifts_cpu] = project_3d_to_2d_back_cpu(
            grad_projections_cpu, reconstruction_cpu, rotations_cpu, shifts_cpu,
            interpolation, oversampling, fourier_radius_cutoff
        );
        
        auto device = grad_projections.device();
        auto grad_rec_result = grad_reconstruction_cpu.to(device, /*non_blocking=*/false);
        auto grad_rot_result = grad_rotations_cpu.numel() > 0 ? grad_rotations_cpu.to(device, /*non_blocking=*/false) : grad_rotations_cpu;
        auto grad_shift_result = grad_shifts_cpu.numel() > 0 ? grad_shifts_cpu.to(device, /*non_blocking=*/false) : grad_shifts_cpu;
        
        return std::make_tuple(grad_rec_result, grad_rot_result, grad_shift_result);
    }

    // Get raw pointers for kernel launch (float32 only)
    const cuFloatComplex* grad_proj_ptr = reinterpret_cast<const cuFloatComplex*>(grad_proj_contiguous.data_ptr<c10::complex<float>>());
    const cuFloatComplex* rec_ptr = reinterpret_cast<const cuFloatComplex*>(rec_contiguous.data_ptr<c10::complex<float>>());
    const float* rot_ptr = rot_contiguous.data_ptr<float>();
    const float* shift_ptr = shifts.has_value() ? shifts_contiguous->data_ptr<float>() : nullptr;
    cuFloatComplex* grad_rec_ptr = reinterpret_cast<cuFloatComplex*>(grad_rec_contiguous.data_ptr<c10::complex<float>>());
    float* grad_rot_ptr = need_rotation_grads ? grad_rot_contiguous.data_ptr<float>() : nullptr;
    float* grad_shift_ptr = need_shift_grads ? grad_shifts_contiguous.data_ptr<float>() : nullptr;

    project_3d_to_2d_back_kernel<<<gridDim, blockDim, 0, stream>>>(
        grad_proj_ptr, rec_ptr, rot_ptr, shift_ptr, 
        grad_rec_ptr, grad_rot_ptr, grad_shift_ptr, params
    );

    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Copy back if needed
    if (!grad_reconstruction.is_contiguous()) {
        grad_reconstruction.copy_(grad_rec_contiguous);
    }
    if (need_rotation_grads && !grad_rotations.is_contiguous()) {
        grad_rotations.copy_(grad_rot_contiguous);
    }
    if (need_shift_grads && !grad_shifts.is_contiguous()) {
        grad_shifts.copy_(grad_shifts_contiguous);
    }

    return std::make_tuple(grad_reconstruction, grad_rotations, grad_shifts);
}

#endif // USE_CUDA