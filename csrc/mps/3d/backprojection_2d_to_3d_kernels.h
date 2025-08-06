/**
 * MPS Kernels for Differentiable 2D->3D Back-Projection Operations in Fourier Space
 * 
 * This file implements back-projection operations that accumulate 2D projections
 * into 3D reconstructions using Apple's Metal Performance Shaders framework.
 * This is the mathematical adjoint/transpose of 3D->2D forward projection and 
 * follows the Central Slice Theorem.
 * 
 * Key Features:
 * - Back-projection: Accumulate 2D projection data into 3D Fourier reconstruction
 * - Optional weight accumulation for CTF^2 or similar applications
 * - Full gradient support for projections, weights, rotations, and shifts
 * - Uses Metal compute shaders for high performance on Apple Silicon
 * - Follows the Central Slice Theorem in Fourier space
 */

#pragma once

#ifdef __APPLE__

#include <torch/extension.h>
#include <c10/util/Optional.h>

/**
 * Forward pass of 2D->3D back-projection on MPS
 * 
 * Accumulates 2D projection data (and optional weights) into 3D reconstructions.
 * This is the adjoint/transpose operation of 3D->2D forward projection.
 * 
 * @param projections: Complex tensor (B, P, height, width/2+1) - 2D projections in Fourier space
 * @param weights: Optional real tensor (B, P, height, width/2+1) - per-pixel weights 
 * @param rotations: Real tensor (B_rot, P, 3, 3) - 3x3 rotation matrices
 * @param shifts: Optional real tensor (B_shift, P, 2) - translation shifts [x, y]
 * @param interpolation: String - "linear" or "cubic" interpolation method
 * @param oversampling: Double - oversampling factor for coordinates
 * @param fourier_radius_cutoff: Optional double - low-pass filter radius
 * 
 * @return: Tuple of (data_reconstruction, weight_reconstruction)
 *          - data_reconstruction: Complex tensor (B, depth, height, width/2+1) 
 *          - weight_reconstruction: Real tensor (B, depth, height, width/2+1) or empty if no weights
 */
std::tuple<at::Tensor, at::Tensor> backproject_2d_to_3d_forw_mps(
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

/**
 * Backward pass of 2D->3D back-projection on MPS
 * 
 * Computes gradients w.r.t. projections, weights, rotations, and shifts.
 * The key insight is that grad_projections = forward_project_3d_to_2d(grad_data_rec) since
 * 3D->2D forward projection is the mathematical adjoint of 2D->3D back-projection.
 * 
 * @param grad_data_rec: Complex tensor (B, depth, height, width/2+1) - gradient w.r.t. data reconstruction
 * @param grad_weight_rec: Optional real tensor (B, depth, height, width/2+1) - gradient w.r.t. weight reconstruction  
 * @param projections: Complex tensor (B, P, height, width/2+1) - original projections (for gradient computation)
 * @param weights: Optional real tensor (B, P, height, width/2+1) - original weights (for gradient computation)
 * @param rotations: Real tensor (B_rot, P, 3, 3) - original rotation matrices
 * @param shifts: Optional real tensor (B_shift, P, 2) - original shifts
 * @param interpolation: String - interpolation method used in forward pass
 * @param oversampling: Double - oversampling factor used in forward pass
 * @param fourier_radius_cutoff: Optional double - filter radius used in forward pass
 * 
 * @return: Tuple of (grad_projections, grad_weights, grad_rotations, grad_shifts)
 *          - grad_projections: Complex tensor (B, P, height, width/2+1)
 *          - grad_weights: Real tensor (B, P, height, width/2+1) or empty
 *          - grad_rotations: Real tensor (B_rot, P, 3, 3) or empty
 *          - grad_shifts: Real tensor (B_shift, P, 2) or empty
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backproject_2d_to_3d_back_mps(
    const at::Tensor& grad_data_rec,
    const c10::optional<at::Tensor>& grad_weight_rec,
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

#endif // __APPLE__