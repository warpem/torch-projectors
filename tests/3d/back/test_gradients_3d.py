"""
Gradient computation tests for 3D->3D back-projection operations.

This module tests gradient computation including basic gradcheck,
rotation gradients, 3D shift gradients, and comprehensive gradient verification.
"""

import torch
import torch.nn.functional as F
import torch_projectors
import pytest
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

# Add parent directory to path to import test_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from test_utils import (
    device, normalized_cross_correlation, complex_mse_loss
)


def create_rotation_matrix_3d_z(angle):
    """Helper function to create 3D rotation matrix around Z axis from angle"""
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    return torch.stack([
        torch.stack([cos_a, -sin_a, torch.zeros_like(angle)], dim=-1),
        torch.stack([sin_a, cos_a, torch.zeros_like(angle)], dim=-1),
        torch.stack([torch.zeros_like(angle), torch.zeros_like(angle), torch.ones_like(angle)], dim=-1)
    ], dim=-2)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_3d_backward_gradcheck_proj_only(device, interpolation):
    """
    Tests the 3D->3D back-projection's backward pass using gradcheck for projections only.

    Note: This test masks out elements on the c=0 plane that have Friedel symmetric counterparts
    to avoid gradcheck failures due to the inherent asymmetry between forward and backward passes
    in Friedel symmetry handling.
    """

    torch.manual_seed(42)

    # Skip MPS for gradcheck - PyTorch gradcheck doesn't support MPS complex ops yet
    if device.type == "mps":
        pytest.skip("gradcheck not supported for MPS with complex tensors")

    B, P, D, H, W = 1, 1, 8, 8, 8  # Smaller for gradcheck
    W_half = W // 2 + 1
    projections = torch.randn(B, P, D, H, W_half, dtype=torch.complex128, requires_grad=True, device=device)
    rotations = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0).unsqueeze(0)

    # Create a mask to zero out problematic elements on c=0 plane that have Friedel counterparts
    # This avoids gradcheck failures due to the asymmetry in Friedel symmetry handling
    # Match the C++ Friedel skip logic: keep only d < D//2 AND r < H//2 for c=0
    mask = torch.ones_like(projections)
    for i in range(D):
        for j in range(H):
            # Skip Friedel-symmetric half: zero out elements where i >= D//2 OR j >= H//2 on c=0 plane
            if i >= D//2 or j >= H//2:
                mask[0, 0, i, j, 0] = 0

    def func(proj):
        # Apply mask to zero out problematic elements
        proj_masked = proj * mask

        # Back-project and return only data reconstruction
        data_rec, _ = torch_projectors.backproject_3d_forw(
            proj_masked,
            rotations,
            interpolation=interpolation
        )
        return data_rec

    # Use relaxed tolerances for 3D due to accumulation precision in atomic operations
    assert torch.autograd.gradcheck(func, projections, atol=1e-3, rtol=0.1)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("shift_values,batch_size,num_poses", [
    # Basic positive shifts (3D)
    ([2.0, 1.0, 1.5], 1, 1),
    # Negative shifts
    ([-2.0, -1.0, -1.5], 1, 1),
    # Mixed sign shifts
    ([2.0, -1.0, 1.5], 1, 1),
    ([-2.0, 1.0, -1.5], 1, 1),
    # Zero shifts (identity case)
    ([0.0, 0.0, 0.0], 1, 1),
    # One component zero
    ([2.0, 0.0, 0.0], 1, 1),
    ([0.0, 1.0, 0.0], 1, 1),
    ([0.0, 0.0, 1.5], 1, 1),
    # Large shifts
    ([10.0, 5.0, 7.5], 1, 1),
    # Fractional shifts
    ([0.5, 0.3, 0.2], 1, 1),
    # Asymmetric magnitudes
    ([0.1, 10.0, 0.5], 1, 1),
    # Batching tests - multiple projections
    ([2.0, 1.0, 1.5], 3, 1),
    # Batching tests - multiple poses
    ([2.0, 1.0, 1.5], 1, 4),
    # Batching tests - both dimensions
    ([2.0, 1.0, 1.5], 2, 3),
])
def test_shift_gradient_verification_parametrized_backproject_3d(device, interpolation, shift_values, batch_size, num_poses):
    """
    Parametrized test to verify 3D->3D back-projection shift gradient calculation across various scenarios.
    Tests backpropagation from L2 loss between unshifted and shifted reconstructions.
    """
    torch.manual_seed(42)

    # Test parameters
    D, H, W = 32, 32, 32
    B = batch_size
    P = num_poses
    W_half = W // 2 + 1

    # Create test shifts - broadcast the same shift to all batches and poses for consistency
    test_shift = torch.tensor(shift_values, dtype=torch.float32, device=device)
    test_shift = test_shift.unsqueeze(0).unsqueeze(0).expand(B, P, 3)  # Shape: (B, P, 3)

    # Step 1: Create random 3D projections (one set per batch and pose)
    proj_random_fourier = torch.randn(B, P, D, H, W_half, dtype=torch.complex64, device=device)

    # Apply Friedel symmetry masking for backprojection to avoid finite difference issues
    # For backprojection, mask elements on c=0 plane that have distinct Friedel counterparts
    mask = torch.ones_like(proj_random_fourier)
    for i in range(1, D//2):  # Skip DC and elements >= D//2
        for j in range(1, H//2):  # Skip DC and elements >= H//2
            mask[:, :, i, j, 0] = 0  # Zero out elements that have distinct Friedel counterparts

    proj_random_fourier = proj_random_fourier * mask

    # Identity rotation (same for all batches and poses)
    rot_identity = torch.eye(3, dtype=torch.float32, device=device)
    rotations = rot_identity.unsqueeze(0).unsqueeze(0).expand(B, P, 3, 3)  # Shape: (B, P, 3, 3)

    # Step 2: Generate unshifted back-projection
    rec_unshifted, _ = torch_projectors.backproject_3d_forw(
        proj_random_fourier, rotations, shifts=None,
        interpolation=interpolation
    )

    # Step 3: Generate shifted back-projection using our implementation
    shifts_for_our_impl = test_shift.clone().requires_grad_(True)
    rec_shifted_our_impl, _ = torch_projectors.backproject_3d_forw(
        proj_random_fourier, rotations, shifts=shifts_for_our_impl,
        interpolation=interpolation
    )

    # Step 4: Manually construct phase modulation for ground truth
    shifts_for_manual = test_shift.clone().requires_grad_(True)

    # Create coordinate grids for 3D phase calculation
    kz = torch.arange(D, dtype=torch.float32, device=device)
    kz[kz > D // 2] -= D  # Shift to [-D/2, D/2) range
    ky = torch.arange(H, dtype=torch.float32, device=device)
    ky[ky > H // 2] -= H  # Shift to [-H/2, H/2) range
    kx = torch.arange(W // 2 + 1, dtype=torch.float32, device=device)
    kzz, kyy, kxx = torch.meshgrid(kz, ky, kx, indexing='ij')

    # Manual phase modulation for all batches and poses
    # Apply phase modulation to PROJECTIONS, then backproject (matches 2D test pattern)
    proj_shifted_manual = torch.zeros_like(proj_random_fourier)
    for b in range(B):
        for p in range(P):
            # Manual 3D phase modulation: exp(+2πi(kz*shift_d + ky*shift_r + kx*shift_c)) for back-projection (conjugate)
            # Note: normalization by H to match the implementation
            phase = 2.0 * math.pi * (kzz * shifts_for_manual[b, p, 0] / H +
                                   kyy * shifts_for_manual[b, p, 1] / H +
                                   kxx * shifts_for_manual[b, p, 2] / H)
            phase_factor = torch.complex(torch.cos(phase), torch.sin(phase))

            # Apply manual phase modulation to projections before back-projection
            proj_shifted_manual[b, p] = proj_random_fourier[b, p] * phase_factor

    # Back-project the manually shifted projections
    rec_shifted_manual, _ = torch_projectors.backproject_3d_forw(
        proj_shifted_manual, rotations, shifts=None,
        interpolation=interpolation
    )

    # Step 5: Calculate losses and gradients
    # Our implementation
    loss_our_impl = complex_mse_loss(rec_shifted_our_impl, rec_unshifted)
    loss_our_impl.backward()
    grad_our_impl = shifts_for_our_impl.grad.clone()

    # Manual implementation
    loss_manual = complex_mse_loss(rec_shifted_manual, rec_unshifted)
    loss_manual.backward()
    grad_manual = shifts_for_manual.grad.clone()

    # Step 6: Compare gradients
    grad_diff = torch.abs(grad_our_impl - grad_manual)
    rel_error = grad_diff / (torch.abs(grad_manual) + 1e-8)
    max_abs_error = grad_diff.max().item()
    max_rel_error = rel_error.max().item()

    print(f"3D->3D Backproject Shift: {shift_values}, Batch: {batch_size}, Poses: {num_poses}")
    print(f"Max absolute error: {max_abs_error:.2e}")
    print(f"Max relative error: {max_rel_error:.2e}")

    # Test assertion - gradients should be close
    torch.testing.assert_close(grad_our_impl, grad_manual, atol=1e-4, rtol=1e-2)

    print(f"✅ 3D->3D Backproject shift gradient test passed: {shift_values} (B={batch_size}, P={num_poses})")


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_gradient_fourier_components_backproject_3d(device, interpolation):
    """
    Test for gradient calculation of Fourier components in 3D->3D back-projection.
    Tests that gradients correctly flow back to the 3D projections.
    """
    torch.manual_seed(42)

    # Test parameters
    D, H, W = 32, 32, 32
    B = 1  # Single batch
    P = 1  # Single pose
    W_half = W // 2 + 1

    # Step 1: Create random 3D projection and back-project with identity rotation
    proj_random_real = torch.randn(B, P, D, H, W, device=device)
    proj_random_fourier = torch.fft.rfftn(proj_random_real, dim=(-3, -2, -1))

    # Identity rotation
    rot_identity = torch.eye(3, dtype=torch.float32, device=device)
    rotations = rot_identity.unsqueeze(0).unsqueeze(0)  # Shape: (B=1, P=1, 3, 3)

    # Back-project with identity rotation, no shift
    rec_identity_fourier, _ = torch_projectors.backproject_3d_forw(
        proj_random_fourier, rotations, shifts=None,
        interpolation=interpolation
    )

    # Convert to real space - this is our reference reconstruction
    ref_rec_identity_real = torch.fft.irfftn(rec_identity_fourier.squeeze(), dim=(-3, -2, -1))

    # Step 2: Initialize zero projection with gradients required
    proj_zero_real = torch.zeros(B, P, D, H, W, requires_grad=True, device=device)
    proj_zero_fourier = torch.fft.rfftn(proj_zero_real, dim=(-3, -2, -1))

    # Step 3: Back-project the zero projection
    rec_zero_fourier, _ = torch_projectors.backproject_3d_forw(
        proj_zero_fourier, rotations, shifts=None,
        interpolation=interpolation
    )
    rec_zero_real = torch.fft.irfftn(rec_zero_fourier.squeeze(), dim=(-3, -2, -1))

    # Step 4: Calculate MSE loss and backpropagate
    loss = F.mse_loss(rec_zero_real, ref_rec_identity_real.detach())
    loss.backward()

    # Step 5: Single gradient descent update
    learning_rate = 1.0
    with torch.no_grad():
        if proj_zero_real.grad is not None:
            proj_zero_real -= learning_rate * proj_zero_real.grad

    # Step 6: Calculate normalized cross-correlation with the expected result
    ncc = normalized_cross_correlation(proj_zero_real.squeeze(0).squeeze(0), proj_random_real.squeeze(0).squeeze(0))

    # Step 7: Create visualizations (central slices)
    os.makedirs('test_outputs/3d/back', exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    slice_idx = D // 2

    # Real space plots (central slice)
    axes[0, 0].imshow(proj_random_real.squeeze()[slice_idx].detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 0].set_title('Reference projection (z=D/2)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(proj_zero_real.squeeze()[slice_idx].detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 1].set_title('Updated projection (z=D/2)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(rec_identity_fourier.squeeze()[slice_idx].abs().detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 2].set_title('Reconstruction (z=D/2)')
    axes[0, 2].axis('off')

    # Add text with results
    axes[1, 0].text(0.1, 0.7, f'NCC: {ncc.item():.6f}', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.5, f'Loss: {loss.item():.6f}', fontsize=12, transform=axes[1, 0].transAxes)
    grad_norm = proj_zero_real.grad.norm().item() if proj_zero_real.grad is not None else 0.0
    axes[1, 0].text(0.1, 0.3, f'Grad norm: {grad_norm:.6f}', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')

    axes[1, 1].axis('off')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'test_outputs/3d/back/gradient_debug_fourier_components_backproject_3d_{interpolation}_{device.type}.png')
    plt.close()

    print(f"3D->3D Backproject Normalized cross-correlation: {ncc.item():.6f}")
    print(f"Loss: {loss.item():.6f}")
    grad_norm = proj_zero_real.grad.norm().item() if proj_zero_real.grad is not None else 0.0
    print(f"Gradient norm: {grad_norm:.6f}")

    # The test - NCC should be reasonably high if gradients work correctly
    assert ncc > 0.5, f"Expected NCC > 0.5, got {ncc.item():.6f}"
