#include "cuda_kernels.h"

#ifdef USE_CUDA

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../cpu/cpu_kernels.h"

// CUDA utility functions and kernels

struct CudaParams {
    int B, P, boxsize, boxsize_half;
    int proj_boxsize, proj_boxsize_half, B_rot, B_shift;
    int has_shifts;
    int interpolation_method; // 0=linear, 1=cubic
    float oversampling;
    float fourier_radius_cutoff;
};

// Complex number operations using cuFloatComplex
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

// Bicubic interpolation kernel helper functions
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

// Forward projection CUDA kernel
__global__ void forward_project_2d_kernel(
    const cuFloatComplex* reconstruction,
    const float* rotations,
    const float* shifts,
    cuFloatComplex* projections,
    CudaParams params
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
    
    // Pre-compute constants for this projection
    int proj_batch_stride = params.P * params.proj_boxsize * params.proj_boxsize_half;
    int proj_pose_stride = params.proj_boxsize * params.proj_boxsize_half;
    int proj_row_stride = params.proj_boxsize_half;
    int proj_base_idx = b * proj_batch_stride + p * proj_pose_stride;
    
    int rec_batch_stride = params.boxsize * params.boxsize_half;
    int rec_row_stride = params.boxsize_half;
    
    float fourier_cutoff_sq = params.fourier_radius_cutoff * params.fourier_radius_cutoff;
    
    // Loop over all pixels in this projection, with threads cooperating
    int total_pixels = params.proj_boxsize * params.proj_boxsize_half;
    
    for (int pixel_idx = threadIdx.x; pixel_idx < total_pixels; pixel_idx += blockDim.x) {
        // Convert linear pixel index to (i, j) coordinates
        int i = pixel_idx / params.proj_boxsize_half;  // Row
        int j = pixel_idx % params.proj_boxsize_half;  // Column
        
        // Convert array indices to Fourier coordinates (must match CPU logic exactly)
        float proj_coord_c = float(j);  // Column: always positive (FFTW half-space)
        float proj_coord_r = (i <= params.proj_boxsize / 2) ? float(i) : float(i) - float(params.proj_boxsize);
        
        // Apply Fourier space filtering (low-pass)
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            projections[proj_base_idx + pixel_idx] = make_cuFloatComplex(0.0f, 0.0f);
            continue;
        }
        
        // Apply oversampling scaling to coordinates
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Apply rotation matrix (using pre-computed matrix elements)
        float rot_c = rot_00 * sample_c + rot_01 * sample_r;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r;
        
        // Interpolate from reconstruction at rotated coordinates
        cuFloatComplex val;
        if (params.interpolation_method == 0) {  // linear
            val = bilinear_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                     rec_batch_stride, rec_row_stride, rot_r, rot_c);
        } else {  // cubic
            val = bicubic_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                    rec_batch_stride, rec_row_stride, rot_r, rot_c);
        }
        
        // Apply phase shift if translations are provided (using pre-computed shifts)
        if (params.has_shifts) {
            float phase = -2.0f * M_PI * (proj_coord_r * shift_r / params.boxsize + 
                                          proj_coord_c * shift_c / params.boxsize);
            cuFloatComplex phase_factor = make_cuFloatComplex(cosf(phase), sinf(phase));
            val = complex_mul(val, phase_factor);
        }
        
        projections[proj_base_idx + pixel_idx] = val;
    }
}

// Forward projection from 3D Fourier reconstruction to 2D projections (CUDA version)
at::Tensor forward_project_2d_cuda(
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
    TORCH_CHECK(reconstruction.dim() == 3,
                "Reconstruction must be a 3D tensor (B, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2,
                "Rotations must be (B_rot, P, 2, 2)");

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
    const auto boxsize = reconstruction.size(1);    // Reconstruction height
    const auto boxsize_half = reconstruction.size(2); // Reconstruction width (FFTW format)
    
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
    CudaParams params = {
        (int)B, (int)P, (int)boxsize, (int)boxsize_half,
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

    // Get raw pointers for kernel launch
    const cuFloatComplex* rec_ptr = reinterpret_cast<const cuFloatComplex*>(rec_contiguous.data_ptr<c10::complex<float>>());
    const float* rot_ptr = rot_contiguous.data_ptr<float>();
    const float* shift_ptr = shifts.has_value() ? shifts_contiguous->data_ptr<float>() : nullptr;
    cuFloatComplex* proj_ptr = reinterpret_cast<cuFloatComplex*>(proj_contiguous.data_ptr<c10::complex<float>>());

    forward_project_2d_kernel<<<gridDim, blockDim, 0, stream>>>(
        rec_ptr, rot_ptr, shift_ptr, proj_ptr, params
    );

    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Copy back if projection wasn't originally contiguous
    if (!projection.is_contiguous()) {
        projection.copy_(proj_contiguous);
        return projection;
    }
    
    return proj_contiguous;
}

// Backward projection for gradients (CUDA version) 
std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_project_2d_cuda(
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
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2,
                "Rotations must be (B_rot, P, 2, 2)");

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

    // Set CUDA device guard
    const c10::cuda::CUDAGuard device_guard(grad_projections.device());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // TEMPORARY: Use CPU implementation as fallback while maintaining gradient flow
    // TODO: Replace with actual CUDA kernel implementation
    
    auto grad_projections_cpu = grad_projections.cpu();
    auto reconstruction_cpu = reconstruction.cpu();
    auto rotations_cpu = rotations.cpu();
    c10::optional<at::Tensor> shifts_cpu;
    if (shifts.has_value()) {
        shifts_cpu = shifts->cpu();
    }
    
    // Call CPU implementation
    auto [grad_reconstruction_cpu, grad_rotations_cpu, grad_shifts_cpu] = backward_project_2d_cpu(
        grad_projections_cpu,
        reconstruction_cpu,
        rotations_cpu,
        shifts_cpu,
        interpolation,
        oversampling,
        fourier_radius_cutoff
    );
    
    // Transfer results back to CUDA with proper device handling
    auto device = grad_projections.device();
    auto grad_reconstruction = grad_reconstruction_cpu.to(device, /*non_blocking=*/false);
    auto grad_rotations = grad_rotations_cpu.numel() > 0 ? grad_rotations_cpu.to(device, /*non_blocking=*/false) : grad_rotations_cpu;
    auto grad_shifts = grad_shifts_cpu.numel() > 0 ? grad_shifts_cpu.to(device, /*non_blocking=*/false) : grad_shifts_cpu;
    
    return std::make_tuple(grad_reconstruction, grad_rotations, grad_shifts);
}

#endif // USE_CUDA