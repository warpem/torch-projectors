/**
 * Header for CPU Kernels for Differentiable 3D->3D Projection Operations in Fourier Space
 *
 * This file declares high-performance CPU kernels for forward and backward projection
 * operations that project 3D Fourier volumes to 3D projections. Used in cryo-electron
 * tomography subtomogram averaging and related 3D imaging applications.
 *
 * Key Concepts:
 * - Forward projection: Sample from 4D Fourier reconstruction [B,D,H,W/2+1] to create 3D projections
 * - Backward projection: Scatter 3D projection gradients into 4D Fourier reconstruction
 * - 3D rotation: Full 3D rotation maps (c,r,d) → rotated coordinates in 3D volume
 * - 3D Friedel symmetry: F(kx,ky,kz) = conj(F(-kx,-ky,-kz)) for real-valued reconstructions
 * - Interpolation: Trilinear (8 neighbors) and tricubic (64 neighbors) with analytical gradients
 */

#pragma once

#include <torch/extension.h>
#include <c10/util/Optional.h>

/**
 * Forward projection from 4D Fourier reconstruction to 3D projections (CPU implementation)
 *
 * Projects 3D Fourier-space volumes to 3D projections using rotation and optional shifts.
 * Each 3D output voxel (i,j,k) corresponds to sampling the 3D volume at rotated coordinates
 * derived from applying the rotation: (c,r,d) → R * (c,r,d).
 *
 * Algorithm:
 * 1. For each output voxel (i,j,k) in the 3D projection
 * 2. Convert to 3D Fourier coordinates (d, r, c)
 * 3. Apply oversampling if specified: (d_sample, r_sample, c_sample)
 * 4. Apply 3x3 rotation matrix: [rot_c, rot_r, rot_d] = R * [c_sample, r_sample, d_sample]
 * 5. Interpolate from 4D reconstruction using trilinear or tricubic interpolation
 * 6. Apply 3D phase shift if translations are provided
 * 7. Store result in 3D projection
 *
 * @param reconstruction: 4D complex tensor [B, D, H, W/2+1] in FFTW format
 * @param rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices
 * @param shifts: Optional 3D real tensor [B_shift, P, 3] - 3D translation shifts
 * @param output_shape: Shape of output projections [D_out, H_out, W_out]
 * @param interpolation: "linear" (trilinear) or "cubic" (tricubic)
 * @param oversampling: Coordinate scaling factor (>1 for oversampling)
 * @param fourier_radius_cutoff: Optional frequency cutoff for low-pass filtering
 * @return: 5D complex tensor [B, P, D_out, H_out, W_out/2+1] - the 3D projections
 */
at::Tensor project_3d_forw_cpu(
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef output_shape,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

/**
 * Unified backward projection function for 3D->3D projections (CPU implementation)
 *
 * Computes gradients w.r.t. reconstruction, rotations, and shifts based on what
 * requires gradients. This implements the adjoint operations of the forward projection,
 * ensuring mathematical consistency for optimization and machine learning applications.
 *
 * Features:
 * 1. Always computes reconstruction gradients (main scatter operation from 3D to 4D)
 * 2. Only computes rotation gradients if rotations.requires_grad() is true
 * 3. Only computes shift gradients if shifts exist and require gradients
 * 4. Uses proper adjoint interpolation operations for mathematical consistency
 *
 * Algorithm:
 * 1. For each projection voxel with incoming gradient
 * 2. Compute corresponding 3D sampling coordinates (same as forward pass)
 * 3. Distribute gradient to 3D reconstruction using adjoint interpolation
 * 4. If needed, compute gradients w.r.t. rotation matrix elements using chain rule
 * 5. If needed, compute gradients w.r.t. 3D shift parameters using phase derivatives
 *
 * @param grad_projections: 5D complex tensor [B, P, D, H, W/2+1] - incoming gradients
 * @param reconstruction: 4D complex tensor [B, D, H, W/2+1] - 3D Fourier volume
 * @param rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices
 * @param shifts: Optional 3D real tensor [B_shift, P, 3] - 3D translation shifts
 * @param interpolation: "linear" (trilinear) or "cubic" (tricubic)
 * @param oversampling: Coordinate scaling factor (must match forward pass)
 * @param fourier_radius_cutoff: Optional frequency cutoff (must match forward pass)
 * @return: Tuple of (grad_reconstruction, grad_rotations, grad_shifts)
 *          Empty tensors are returned for gradients that aren't needed
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> project_3d_back_cpu(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);
