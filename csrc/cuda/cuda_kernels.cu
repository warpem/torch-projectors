#include "cuda_kernels.h"

#ifdef USE_CUDA

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../cpu/cpu_kernels.h"

// CUDA kernel declarations (placeholders for future implementation)
// TODO: Replace CPU fallback with actual CUDA kernels

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

    // Set CUDA device guard to ensure operations happen on the right device
    const c10::cuda::CUDAGuard device_guard(reconstruction.device());
    
    // Get CUDA stream for asynchronous operations
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // TEMPORARY: Use CPU implementation as fallback while maintaining gradient flow
    // TODO: Replace with actual CUDA kernel implementation 
    // 
    // To preserve gradients, we use a custom autograd function that handles
    // the CPU fallback while keeping tensors on CUDA device in the autograd graph
    
    // For now, transfer to CPU, compute, and transfer back
    // This maintains the gradient flow by keeping the operation in CUDA context
    auto reconstruction_cpu = reconstruction.cpu();
    auto rotations_cpu = rotations.cpu();
    c10::optional<at::Tensor> shifts_cpu;
    if (shifts.has_value()) {
        shifts_cpu = shifts->cpu();
    }
    
    // Call CPU implementation
    auto result_cpu = forward_project_2d_cpu(
        reconstruction_cpu,
        rotations_cpu, 
        shifts_cpu,
        output_shape,
        interpolation,
        oversampling,
        fourier_radius_cutoff
    );
    
    // Transfer result back to CUDA and ensure it maintains gradient tracking
    auto result_cuda = result_cpu.to(reconstruction.device(), /*non_blocking=*/false);
    
    // Explicitly maintain gradient tracking by copying requires_grad state
    if (reconstruction.requires_grad() || rotations.requires_grad() || 
        (shifts.has_value() && shifts->requires_grad())) {
        result_cuda.requires_grad_(true);
    }
    
    return result_cuda;
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