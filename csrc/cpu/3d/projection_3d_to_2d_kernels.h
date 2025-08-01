/**
 * Header for CPU Kernels for Differentiable 3D->2D Projection Operations in Fourier Space
 * 
 * This file declares high-performance CPU kernels for forward and backward projection
 * operations that project 3D Fourier volumes to 2D projections. Used in cryo-electron
 * tomography and related 3D imaging applications.
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

/**
 * Forward projection from 4D Fourier reconstruction to 2D projections (CPU implementation)
 * 
 * Projects 3D Fourier-space volumes to 2D projections using the central slice theorem.
 * Each 2D output pixel (i,j) corresponds to sampling the 3D volume at rotated coordinates
 * derived from the central slice: (proj_r, proj_c, 0) → R * (proj_r, proj_c, 0).
 * 
 * Algorithm:
 * 1. For each output pixel (i,j) in the 2D projection
 * 2. Convert to 2D Fourier coordinates (proj_coord_r, proj_coord_c) 
 * 3. Extend to 3D central slice: (proj_coord_r, proj_coord_c, 0)
 * 4. Apply 3x3 rotation matrix to get sampling coordinates in 3D reconstruction
 * 5. Apply oversampling scaling if specified
 * 6. Interpolate from 4D reconstruction using trilinear or tricubic interpolation
 * 7. Apply phase shift if translations are provided
 * 8. Store result in 2D projection
 * 
 * @param reconstruction: 4D complex tensor [B, D, H, W/2+1] in FFTW format
 * @param rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices
 * @param shifts: Optional 3D real tensor [B_shift, P, 2] - 2D translation shifts
 * @param output_shape: Shape of output projections [H_out, W_out]
 * @param interpolation: "linear" (trilinear) or "cubic" (tricubic)
 * @param oversampling: Coordinate scaling factor (>1 for oversampling)
 * @param fourier_radius_cutoff: Optional frequency cutoff for low-pass filtering
 * @return: 4D complex tensor [B, P, H_out, W_out/2+1] - the 2D projections
 */
at::Tensor forward_project_3d_to_2d_cpu(
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef output_shape,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

/**
 * Unified backward projection function for 3D->2D projections (CPU implementation)
 * 
 * Computes gradients w.r.t. reconstruction, rotations, and shifts based on what
 * requires gradients. This implements the adjoint operations of the forward projection,
 * ensuring mathematical consistency for optimization and machine learning applications.
 * 
 * Features:
 * 1. Always computes reconstruction gradients (main scatter operation from 2D to 4D)
 * 2. Only computes rotation gradients if rotations.requires_grad() is true
 * 3. Only computes shift gradients if shifts exist and require gradients
 * 4. Uses proper adjoint interpolation operations for mathematical consistency
 * 
 * Algorithm:
 * 1. For each projection pixel with incoming gradient
 * 2. Compute corresponding 3D sampling coordinates (same as forward pass)
 * 3. Distribute gradient to 3D reconstruction using adjoint interpolation
 * 4. If needed, compute gradients w.r.t. rotation matrix elements using chain rule
 * 5. If needed, compute gradients w.r.t. shift parameters using phase derivatives
 * 
 * @param grad_projections: 4D complex tensor [B, P, H, W/2+1] - incoming gradients
 * @param reconstruction: 4D complex tensor [B, D, H, W/2+1] - 3D Fourier volume  
 * @param rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices
 * @param shifts: Optional 3D real tensor [B_shift, P, 2] - 2D translation shifts
 * @param interpolation: "linear" (trilinear) or "cubic" (tricubic)
 * @param oversampling: Coordinate scaling factor (must match forward pass)
 * @param fourier_radius_cutoff: Optional frequency cutoff (must match forward pass)
 * @return: Tuple of (grad_reconstruction, grad_rotations, grad_shifts)
 *          Empty tensors are returned for gradients that aren't needed
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_project_3d_to_2d_cpu(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);