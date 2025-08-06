/**
 * CPU Kernels for Differentiable 2D->3D Back-Projection Operations in Fourier Space
 * 
 * This header declares back-projection operations that accumulate 2D projections
 * into 3D reconstructions. This is the mathematical adjoint/transpose of 3D->2D
 * forward projection.
 * 
 * Key Features:
 * - Back-projection: Accumulate 2D projection data into 3D Fourier reconstruction
 * - Full gradient support for projections, rotations, and shifts
 * - Uses shared interpolation kernels, FFTW sampling, and atomic operations
 * - Follows the Central Slice Theorem in Fourier space
 */

#pragma once

#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <tuple>
#include <string>

/**
 * Forward pass: Back-projection from 2D projections to 3D reconstructions
 * 
 * Accumulates 2D projection data (and optional weights) into 3D reconstructions using the Central Slice Theorem.
 * This is the adjoint/transpose operation of 3D->2D forward projection.
 * 
 * @param projections: Input 2D projections (B, P, H, W/2+1) in FFTW format
 * @param weights: Optional weights (B, P, H, W/2+1) for CTF^2 or similar applications
 * @param rotations: 3x3 rotation matrices (B_rot, P, 3, 3)
 * @param shifts: Optional 2D shifts (B_shift, P, 2)
 * @param interpolation: "linear" or "cubic"
 * @param oversampling: Oversampling factor for coordinate scaling
 * @param fourier_radius_cutoff: Optional frequency cutoff radius
 * @return: Tuple of (data_reconstruction, weight_reconstruction) - both 3D (B, D, H, W/2+1)
 */
std::tuple<at::Tensor, at::Tensor> backproject_2d_to_3d_forw_cpu(
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

/**
 * Backward pass: Gradient computation for 2D->3D back-projection
 * 
 * Computes gradients w.r.t. projections, weights, rotations, and shifts.
 * 
 * @param grad_data_rec: Gradient w.r.t. 3D data reconstruction (B, D, H, W/2+1)
 * @param grad_weight_rec: Optional gradient w.r.t. 3D weight reconstruction (B, D, H, W/2+1)
 * @param projections: Original 2D projections (B, P, H, W/2+1)
 * @param weights: Optional weights (B, P, H, W/2+1)
 * @param rotations: 3x3 rotation matrices (B_rot, P, 3, 3)
 * @param shifts: Optional 2D shifts (B_shift, P, 2)
 * @param interpolation: "linear" or "cubic"
 * @param oversampling: Oversampling factor
 * @param fourier_radius_cutoff: Optional frequency cutoff radius
 * @return: Tuple of (grad_projections, grad_weights, grad_rotations, grad_shifts)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backproject_2d_to_3d_back_cpu(
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