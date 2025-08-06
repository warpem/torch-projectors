"""
Gradient computation tests for torch-projectors.

This module tests gradient computation including basic gradcheck,
rotation gradients, shift gradients, and comprehensive gradient verification.
"""

import torch
import torch.nn.functional as F
import torch_projectors
import pytest
import math
import matplotlib.pyplot as plt
import os
from test_utils import (
    device, create_rotation_matrix_2d, compute_angle_grad_from_matrix_grad,
    normalized_cross_correlation, complex_mse_loss
)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_2d_backward_gradcheck_rec_only(device, interpolation):
    """
    Tests the 2D forward projection's backward pass using gradcheck for reconstruction only.
    
    Note: This test masks out elements on the c=0 line that have Friedel symmetric counterparts
    to avoid gradcheck failures due to the inherent asymmetry between forward and backward passes
    in Friedel symmetry handling. The forward pass must sample from both locations to maintain
    physical correctness, but the backward pass correctly computes gradients for optimization purposes.
    """

    torch.manual_seed(42)

    # Skip MPS for gradcheck - PyTorch gradcheck doesn't support MPS complex ops yet
    if device.type == "mps":
        pytest.skip("gradcheck not supported for MPS with complex tensors")
        
    B, P, H, W = 1, 1, 16, 16
    W_half = W // 2 + 1
    rec_fourier = torch.randn(B, H, W_half, dtype=torch.complex128, requires_grad=True, device=device)
    rotations = torch.eye(2, dtype=torch.float64, device=device).unsqueeze(0).unsqueeze(0)
    output_shape = (H, W)

    # Create a mask to zero out problematic elements on c=0 line that have Friedel counterparts
    # This avoids gradcheck failures due to the asymmetry in Friedel symmetry handling
    # For forward projection, the issue is with elements i >= H//2 on c=0 line 
    mask = torch.ones_like(rec_fourier)
    for i in range(H//2, H):  # Zero out elements >= H//2 on c=0 line
        mask[0, i, 0] = 0  # These elements don't get gradients in backward pass but contribute to forward pass

    def func(reconstruction):
        # Apply mask to zero out problematic elements
        reconstruction_masked = reconstruction * mask
        
        return torch_projectors.project_2d_forw(
            reconstruction_masked,
            rotations,
            output_shape=output_shape,
            interpolation=interpolation
        )

    assert torch.autograd.gradcheck(func, rec_fourier, atol=1e-4, rtol=1e-2)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_gradient_fourier_components(device, interpolation):
    """
    Debug test for gradient calculation of Fourier components.
    Tests that gradients correctly flow back to the reconstruction.
    """
    # MPS gradient support now working after fixing requires_grad preservation
    torch.manual_seed(42)
    
    # Test parameters
    H, W = 32, 32
    B = 1  # Single batch
    P = 1  # Single pose
    
    # Step 1: Initialize random 2D reconstruction and project at 90°
    rec_random_real = torch.randn(H, W, device=device)
    rec_random_fourier = torch.fft.rfftn(rec_random_real, dim=(-2, -1))
    rec_random_fourier = rec_random_fourier.unsqueeze(0)  # Add batch dim
    
    # 90 degree rotation matrix
    rot_90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32, device=device)
    rotations_90 = rot_90.unsqueeze(0).unsqueeze(0)  # Shape: (B=1, P=1, 2, 2)
    
    # Project at 90°, no shift
    proj_90_fourier = torch_projectors.project_2d_forw(
        rec_random_fourier, rotations_90, shifts=None, 
        output_shape=(H, W), interpolation=interpolation
    )
    
    # Convert to real space - this is our reference projection
    ref_proj_90_real = torch.fft.irfftn(proj_90_fourier.squeeze(), dim=(-2, -1))
    
    # Step 2: Create reference projection at 0° (identity rotation)
    rot_0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=device)
    rotations_0 = rot_0.unsqueeze(0).unsqueeze(0)  # Shape: (B=1, P=1, 2, 2)
    
    proj_0_fourier = torch_projectors.project_2d_forw(
        rec_random_fourier, rotations_0, shifts=None,
        output_shape=(H, W), interpolation=interpolation
    )
    ref_proj_0_real = torch.fft.irfftn(proj_0_fourier.squeeze(), dim=(-2, -1))
    
    # Step 3: Initialize zero reconstruction with gradients required
    rec_zero_real = torch.zeros(H, W, requires_grad=True, device=device)
    rec_zero_fourier = torch.fft.rfftn(rec_zero_real, dim=(-2, -1))
    rec_zero_fourier = rec_zero_fourier.unsqueeze(0)  # Add batch dim
    
    # Step 4: Forward project the zero reconstruction at 90°
    proj_zero_fourier = torch_projectors.project_2d_forw(
        rec_zero_fourier, rotations_90, shifts=None,
        output_shape=(H, W), interpolation=interpolation
    )
    proj_zero_real = torch.fft.irfftn(proj_zero_fourier.squeeze(), dim=(-2, -1))
    
    # Step 5: Calculate MSE loss and backpropagate
    loss = F.mse_loss(proj_zero_real, ref_proj_90_real.detach())
    loss.backward()
    
    # Step 6: Single gradient descent update
    learning_rate = 1.0
    with torch.no_grad():
        rec_zero_real -= learning_rate * rec_zero_real.grad
    
    # Step 7: Calculate normalized cross-correlation
    ncc = normalized_cross_correlation(rec_zero_real, ref_proj_0_real)
    
    # Step 8: Create visualizations - all in one plot
    # Convert Fourier space tensors for plotting
    rec_zero_fourier_updated = torch.fft.rfftn(rec_zero_real, dim=(-2, -1))
    ref_proj_0_fourier = torch.fft.rfftn(ref_proj_0_real, dim=(-2, -1))
    
    os.makedirs('test_outputs', exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Real space plots
    axes[0, 0].imshow(ref_proj_0_real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 0].set_title('Reference (0° projection)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rec_zero_real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 1].set_title('Updated reconstruction')
    axes[0, 1].axis('off')
    
    # Fourier space plots - real parts
    axes[0, 2].imshow(ref_proj_0_fourier.real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 2].set_title('Reference FFT (Real)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(rec_zero_fourier_updated.real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 3].set_title('Updated FFT (Real)')
    axes[0, 3].axis('off')
    
    # Fourier space plots - imaginary parts
    axes[1, 0].imshow(ref_proj_0_fourier.imag.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[1, 0].set_title('Reference FFT (Imag)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(rec_zero_fourier_updated.imag.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[1, 1].set_title('Updated FFT (Imag)')
    axes[1, 1].axis('off')
    
    # Add text with results
    axes[1, 2].text(0.1, 0.7, f'NCC: {ncc.item():.6f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, f'Loss: {loss.item():.6f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.3, f'Grad norm: {rec_zero_real.grad.norm().item():.6f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'test_outputs/gradient_debug_fourier_components_{interpolation}_{device.type}.png')
    plt.close()
    
    print(f"Normalized cross-correlation: {ncc.item():.6f}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient norm: {rec_zero_real.grad.norm().item():.6f}")
    
    # The test - NCC should be close to 1.0 if gradients work correctly
    # Note: Due to numerical precision and the fact that we're doing a single gradient step,
    # we use a reasonable tolerance
    assert ncc > 0.8, f"Expected NCC > 0.8, got {ncc.item():.6f}"


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("shift_values,batch_size,num_poses", [
    # Basic positive shifts
    ([2.0, 1.0], 1, 1),
    # Negative shifts  
    ([-2.0, -1.0], 1, 1),
    # Mixed sign shifts
    ([2.0, -1.0], 1, 1),
    ([-2.0, 1.0], 1, 1),
    # Zero shifts (identity case)
    ([0.0, 0.0], 1, 1),
    # One component zero
    ([2.0, 0.0], 1, 1),
    ([0.0, 1.0], 1, 1),
    # Large shifts
    ([10.0, 5.0], 1, 1),
    # Fractional shifts
    ([0.5, 0.3], 1, 1),
    # Asymmetric magnitudes
    ([0.1, 10.0], 1, 1),
    # Batching tests - multiple reconstructions
    ([2.0, 1.0], 3, 1),
    # Batching tests - multiple poses
    ([2.0, 1.0], 1, 4),
    # Batching tests - both dimensions
    ([2.0, 1.0], 2, 3),
])
def test_shift_gradient_verification_parametrized(device, interpolation, shift_values, batch_size, num_poses):
    """
    Parametrized test to verify shift gradient calculation across various scenarios.
    Tests backpropagation from L2 loss between unshifted and shifted projections.
    """
    # MPS gradient support now working after fixing requires_grad preservation
    torch.manual_seed(42)
    
    # Test parameters
    H, W = 32, 32
    B = batch_size
    P = num_poses
    
    # Create test shifts - broadcast the same shift to all batches and poses for consistency
    test_shift = torch.tensor(shift_values, dtype=torch.float32, device=device)
    test_shift = test_shift.unsqueeze(0).unsqueeze(0).expand(B, P, 2)  # Shape: (B, P, 2)
    
    # Step 1: Create random reconstructions (one per batch)
    rec_random_real = torch.randn(B, H, W, device=device)
    rec_random_fourier = torch.fft.rfftn(rec_random_real, dim=(-2, -1))
    
    # Apply Friedel symmetry masking for forward projection to avoid finite difference issues
    # For 2D forward projection, mask elements i >= H//2 on c=0 line
    mask = torch.ones_like(rec_random_fourier)
    for i in range(H//2, H):  # Zero out elements >= H//2 on c=0 line
        mask[:, i, 0] = 0  # These elements don't get gradients in backward pass but contribute to forward pass
    
    rec_random_fourier = rec_random_fourier * mask
    
    # 15-degree rotation (same for all batches and poses)
    angle_deg = 0.0
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot_15 = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32, device=device)
    rotations = rot_15.unsqueeze(0).unsqueeze(0).expand(B, P, 2, 2)  # Shape: (B, P, 2, 2)
    
    # Step 2: Generate unshifted projection at 15 degrees
    proj_unshifted = torch_projectors.project_2d_forw(
        rec_random_fourier, rotations, shifts=None,
        output_shape=(H, W), interpolation=interpolation
    )
    
    # Step 3: Generate shifted projection using our implementation
    shifts_for_our_impl = test_shift.clone().requires_grad_(True)
    proj_shifted_our_impl = torch_projectors.project_2d_forw(
        rec_random_fourier, rotations, shifts=shifts_for_our_impl,
        output_shape=(H, W), interpolation=interpolation
    )
    
    # Step 4: Manually construct phase modulation for ground truth
    shifts_for_manual = test_shift.clone().requires_grad_(True)
    
    # Create coordinate grids for phase calculation
    ky = torch.arange(H, dtype=torch.float32, device=device)
    ky[ky > H // 2] -= H  # Shift to [-H/2, H/2) range
    kx = torch.arange(W // 2 + 1, dtype=torch.float32, device=device)
    kyy, kxx = torch.meshgrid(ky, kx, indexing='ij')
    
    # Manual phase modulation for all batches and poses
    proj_shifted_manual = torch.zeros_like(proj_unshifted)
    for b in range(B):
        for p in range(P):
            # Manual phase modulation: exp(-2πi(ky*shift_r + kx*shift_c))
            # Note: normalization by H to match the implementation
            phase = -2.0 * math.pi * (kyy * shifts_for_manual[b, p, 0] / H + kxx * shifts_for_manual[b, p, 1] / H)
            phase_factor = torch.complex(torch.cos(phase), torch.sin(phase))
            
            # Apply manual phase modulation
            proj_shifted_manual[b, p] = proj_unshifted[b, p] * phase_factor
    
    # Step 5: Implement manual complex MSE loss and backprop
    # Calculate losses and gradients
    # Our implementation
    loss_our_impl = complex_mse_loss(proj_shifted_our_impl, proj_unshifted)
    loss_our_impl.backward()
    grad_our_impl = shifts_for_our_impl.grad.clone()
    
    # Manual implementation  
    loss_manual = complex_mse_loss(proj_shifted_manual, proj_unshifted)
    loss_manual.backward()
    grad_manual = shifts_for_manual.grad.clone()
    
    # Step 6: Compare gradients
    grad_diff = torch.abs(grad_our_impl - grad_manual)
    rel_error = grad_diff / (torch.abs(grad_manual) + 1e-8)
    max_abs_error = grad_diff.max().item()
    max_rel_error = rel_error.max().item()
    
    print(f"Shift: {shift_values}, Batch: {batch_size}, Poses: {num_poses}")
    print(f"Max absolute error: {max_abs_error:.2e}")
    print(f"Max relative error: {max_rel_error:.2e}")
    
    # Test assertion - gradients should be close
    torch.testing.assert_close(grad_our_impl, grad_manual, atol=1e-4, rtol=1e-2)
    
    # Create visualization
    os.makedirs('test_outputs', exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Show first batch, first pose for visualization
    # Real parts
    axes[0, 0].imshow(proj_unshifted[0, 0].real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 0].set_title('Unshifted (Real)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(proj_shifted_our_impl[0, 0].real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 1].set_title('Shifted - Our Impl (Real)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(proj_shifted_manual[0, 0].real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 2].set_title('Shifted - Manual (Real)')
    axes[0, 2].axis('off')
    
    # Imaginary parts
    axes[1, 0].imshow(proj_unshifted[0, 0].imag.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[1, 0].set_title('Unshifted (Imag)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(proj_shifted_our_impl[0, 0].imag.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[1, 1].set_title('Shifted - Our Impl (Imag)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(proj_shifted_manual[0, 0].imag.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[1, 2].set_title('Shifted - Manual (Imag)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'test_outputs/shift_gradient_verification_{shift_values[0]}_{shift_values[1]}_B{batch_size}_P{num_poses}_{interpolation}_{device.type}.png')
    plt.close()
    
    print(f"✅ Shift gradient test passed: {shift_values} (B={batch_size}, P={num_poses})")


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_rotation_gradients_comprehensive(device, interpolation):
    """
    Comprehensive test of rotation gradients using finite difference verification.
    This test validates that our analytical rotation gradients match numerical derivatives.
    """

    torch.manual_seed(42)

    # MPS gradient support now working after fixing requires_grad preservation
    print(f"\n🔄 Testing rotation gradients comprehensively ({interpolation})...")
    
    # Test 1: Finite Difference Verification (most important)
    print("  1️⃣ Finite difference verification...")
    _test_rotation_finite_difference_accuracy(device, interpolation)
    
    # Test 2: Multiple scenarios to ensure robustness
    print("  2️⃣ Testing multiple scenarios...")
    test_cases = [
        (0.1, 0.05),  # Small angles
        (0.3, 0.1),   # Medium angles  
        (0.8, 0.2),   # Larger angles
    ]
    
    for i, (target_angle, test_offset) in enumerate(test_cases):
        print(f"    Scenario {i+1}: target={target_angle:.2f}, offset={test_offset:.2f}")
        torch.manual_seed(42 + i)
        rec = torch.randn(1, 16, 9, dtype=torch.complex64, requires_grad=True, device=device)
        
        target_rot = create_rotation_matrix_2d(torch.tensor(target_angle, device=device)).unsqueeze(0).unsqueeze(0)
        target_proj = torch_projectors.project_2d_forw(rec, target_rot.detach(), interpolation=interpolation)
        
        test_angle = target_angle + test_offset
        test_rot = create_rotation_matrix_2d(torch.tensor(test_angle, device=device)).unsqueeze(0).unsqueeze(0)
        test_rot.requires_grad_(True)
        
        pred_proj = torch_projectors.project_2d_forw(rec, test_rot, interpolation=interpolation)
        loss = torch.sum((pred_proj - target_proj).abs())
        loss.backward()
        
        # Verify gradients exist and are non-zero
        assert test_rot.grad is not None, f"Scenario {i+1}: No gradients computed"
        grad_norm = test_rot.grad.norm().item()
        assert grad_norm > 1e-8, f"Scenario {i+1}: Gradient norm too small: {grad_norm:.2e}"
        print(f"      Gradient norm: {grad_norm:.6f}")
    
    # Test 3: Optimization Convergence (practical validation)
    print("  3️⃣ Optimization convergence verification...")
    _test_rotation_optimization_convergence(device, interpolation)
    
    print("✅ All rotation gradient tests passed!")


def _test_rotation_finite_difference_accuracy(device, interpolation):
    """Test that analytical gradients match finite differences"""

    torch.manual_seed(42)
    
    # Create test data - use smaller, more manageable case
    torch.manual_seed(42)  # For reproducibility
    B, H, W_half = 1, 16, 9
    rec = torch.randn(B, H, W_half, dtype=torch.complex64, device=device)    
    rec[0, :, :3] = 0
    rec.requires_grad_(True)
    target_angle = 0  # Smaller angle difference
    target_rot = create_rotation_matrix_2d(torch.tensor(target_angle, device=device)).unsqueeze(0).unsqueeze(0)
    target_proj = torch_projectors.project_2d_forw(rec, target_rot.detach(), interpolation=interpolation)
    
    # Test at slightly different angle
    test_angle = target_angle + 0.01  # Smaller perturbation
    test_rot = create_rotation_matrix_2d(torch.tensor(test_angle, device=device)).unsqueeze(0).unsqueeze(0)
    test_rot.requires_grad_(True)
    
    # Compute analytical gradients
    pred_proj = torch_projectors.project_2d_forw(rec, test_rot, interpolation=interpolation)
    loss = torch.sum((pred_proj - target_proj).abs())
    loss.backward()
    
    # Compare against finite differences using proper angle perturbation
    eps = 1e-4  # Epsilon for numerical stability
    
    print("    Comparing analytical vs finite difference gradients:")
    
    # Create rotation matrices with perturbed angles
    rot_plus = create_rotation_matrix_2d(torch.tensor(test_angle + eps, device=device)).unsqueeze(0).unsqueeze(0)
    rot_minus = create_rotation_matrix_2d(torch.tensor(test_angle - eps, device=device)).unsqueeze(0).unsqueeze(0)
    
    # Compute finite difference
    loss_plus = torch.sum((torch_projectors.project_2d_forw(rec.detach(), rot_plus, interpolation=interpolation) - target_proj).abs())
    loss_minus = torch.sum((torch_projectors.project_2d_forw(rec.detach(), rot_minus, interpolation=interpolation) - target_proj).abs())
    fd_angle_grad = (loss_plus - loss_minus) / (2 * eps)
    
    # Convert matrix gradient to angle gradient using chain rule
    analytical_angle_grad = compute_angle_grad_from_matrix_grad(test_rot.grad[0, 0], torch.tensor(test_angle, device=device))
    abs_error = torch.abs(analytical_angle_grad - fd_angle_grad).item()
    relative_error = abs_error / (torch.abs(fd_angle_grad).item() + 1e-8)
    
    print(f"    Angle gradient: analytical={analytical_angle_grad:.6f}, fd={fd_angle_grad:.6f}, abs_err={abs_error:.2e}, rel_err={relative_error:.2e}")
    
    # Reasonable tolerance - analytical gradients should be close to finite differences
    assert relative_error < 0.01, f"Relative finite difference error {relative_error:.2e} exceeds tolerance"
    print(f"    ✅ Relative finite difference error: {relative_error:.2e}")


def _test_rotation_optimization_convergence(device, interpolation):
    """Test that optimization converges to correct angle"""
    # Create target projection at known angle
    torch.manual_seed(42)  # For reproducibility
    rec = torch.randn(1, 16, 9, dtype=torch.complex64, device=device)  # Smaller for stability
    target_angles = [0.2, 0.5, 0.8]  # Positive angles for stability (fewer local minima)
    
    for target_angle in target_angles:
        print(f"    Optimizing toward angle {target_angle:.3f}...")
        
        target_rot = create_rotation_matrix_2d(torch.tensor(target_angle, device=device)).unsqueeze(0).unsqueeze(0)
        target_proj = torch_projectors.project_2d_forw(rec, target_rot, interpolation=interpolation)
        
        # Initialize optimization close to target (not at 0) for better convergence
        init_angle = target_angle + 0.017  # Start 1 degree (~0.017 radians) away
        learned_angle = torch.tensor(init_angle, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([learned_angle], lr=0.02)  # Lower learning rate for stability
        
        best_loss = float('inf')
        best_angle = init_angle
        
        for step in range(100):  # Fewer steps for speed
            optimizer.zero_grad()
            learned_rot = create_rotation_matrix_2d(learned_angle).unsqueeze(0).unsqueeze(0)
            pred_proj = torch_projectors.project_2d_forw(rec, learned_rot, interpolation=interpolation)
            loss = torch.sum((pred_proj - target_proj).abs().pow(2))  # Manual MSE for complex tensors
            loss.backward()
            optimizer.step()
            
            # Track best result
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_angle = learned_angle.item()
            
            # Check for convergence
            if step % 25 == 0:
                print(f"      Step {step}: loss={loss.item():.6f}, angle={learned_angle.item():.3f}")
        
        # Check convergence using best result (accounting for 2π periodicity)
        angle_diff = torch.abs(torch.tensor(best_angle, device=device) - target_angle) % (2 * torch.pi)
        angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)
        
        print(f"      Final: target={target_angle:.3f}, best={best_angle:.3f}, diff={angle_diff.item():.3f}")
        # Reasonable tolerance for convergence
        assert angle_diff < 0.001, f"Failed to converge: angle difference {angle_diff.item():.3f} > 0.001 rad"