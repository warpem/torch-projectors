/**
 * Header for MPS Kernels for Differentiable 3D->2D Projection Operations in Fourier Space
 * 
 * This file declares high-performance MPS (Metal Performance Shaders) kernels for forward 
 * and backward projection operations that project 3D Fourier volumes to 2D projections.
 * Used in cryo-electron tomography and related 3D imaging applications.
 * 
 * Key Concepts:
 * - Forward projection: Sample from 4D Fourier reconstruction [B,D,H,W/2+1] to create 2D projections
 * - Backward projection: Scatter 2D projection gradients into 4D Fourier reconstruction
 * - Central slice projection: 3D rotation maps (r,c,0) → rotated coordinates in 3D volume
 * - 3D Friedel symmetry: F(kx,ky,kz) = conj(F(-kx,-ky,-kz)) for real-valued reconstructions
 * - Interpolation: Trilinear (8 neighbors) and tricubic (64 neighbors) with analytical gradients
 */

#pragma once

#include <torch/extension.h>
#include <c10/util/Optional.h>

#ifdef __APPLE__

/**
 * Forward projection from 4D Fourier reconstruction to 2D projections (MPS implementation)
 * 
 * Projects 3D Fourier-space volumes to 2D projections using the central slice theorem.
 * Each 2D output pixel (i,j) corresponds to sampling the 3D volume at rotated coordinates
 * derived from the central slice: (proj_r, proj_c, 0) → R * (proj_r, proj_c, 0).
 * 
 * @param reconstruction: 4D complex tensor [B, D, H, W/2+1] in FFTW format (on MPS device)
 * @param rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices (on MPS device)
 * @param shifts: Optional 3D real tensor [B_shift, P, 2] - 2D translation shifts (on MPS device)
 * @param output_shape: Shape of output projections [H_out, W_out]
 * @param interpolation: "linear" (trilinear) or "cubic" (tricubic)
 * @param oversampling: Coordinate scaling factor (>1 for oversampling)
 * @param fourier_radius_cutoff: Optional frequency cutoff for low-pass filtering
 * @return: 4D complex tensor [B, P, H_out, W_out/2+1] - the 2D projections (on MPS device)
 */
at::Tensor project_3d_to_2d_forw_mps(
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef output_shape,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

/**
 * Unified backward projection function for 3D->2D projections (MPS implementation)
 * 
 * Computes gradients w.r.t. reconstruction, rotations, and shifts based on what
 * requires gradients. This implements the adjoint operations of the forward projection,
 * ensuring mathematical consistency for optimization and machine learning applications.
 * 
 * @param grad_projections: 4D complex tensor [B, P, H, W/2+1] - incoming gradients (on MPS device)
 * @param reconstruction: 4D complex tensor [B, D, H, W/2+1] - 3D Fourier volume (on MPS device)
 * @param rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices (on MPS device)
 * @param shifts: Optional 3D real tensor [B_shift, P, 2] - 2D translation shifts (on MPS device)
 * @param interpolation: "linear" (trilinear) or "cubic" (tricubic)
 * @param oversampling: Coordinate scaling factor (must match forward pass)
 * @param fourier_radius_cutoff: Optional frequency cutoff (must match forward pass)
 * @return: Tuple of (grad_reconstruction, grad_rotations, grad_shifts) (all on MPS device)
 *          Empty tensors are returned for gradients that aren't needed
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> project_3d_to_2d_back_mps(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

#endif // __APPLE__