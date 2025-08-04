"""
Gradient computation tests for torch-projectors back-projection.

This module tests gradient computation for back-projection including basic gradcheck,
rotation gradients, shift gradients, and comprehensive gradient verification.
"""

import torch
import torch.nn.functional as F
import torch_projectors
import pytest
import math
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import test_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_utils import (
    device, create_rotation_matrix_2d, compute_angle_grad_from_matrix_grad,
    normalized_cross_correlation, complex_mse_loss
)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_2d_gradcheck_projections_only(device, interpolation):
    """
    Tests the 2D back-projection's backward pass using gradcheck for projections only.
    """

    torch.manual_seed(42)

    # Skip MPS for gradcheck - PyTorch gradcheck doesn't support MPS complex ops yet
    if device.type == "mps":
        pytest.skip("gradcheck not supported for MPS with complex tensors")
        
    B, P, H, W = 1, 1, 16, 16
    W_half = W // 2 + 1
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex128, requires_grad=True, device=device)
    rotations = torch.eye(2, dtype=torch.float64, device=device).unsqueeze(0).unsqueeze(0)

    def func(projections):
        reconstruction, _ = torch_projectors.backproject_2d_forw(
            projections,
            rotations=rotations,
            interpolation=interpolation
        )
        return reconstruction

    assert torch.autograd.gradcheck(func, projections, atol=1e-4, rtol=1e-2)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_gradient_fourier_components(device, interpolation):
    """
    Debug test for gradient calculation of back-projection Fourier components.
    Tests that gradients correctly flow back to the projections.
    """
    torch.manual_seed(42)
    
    # Test parameters
    H, W = 32, 32
    B = 1  # Single batch
    P = 1  # Single pose
    
    # Step 1: Create reference reconstruction from projections at 90°
    projections_real = torch.randn(H, W, device=device)
    projections_fourier = torch.fft.rfftn(projections_real, dim=(-2, -1))
    projections_fourier = projections_fourier.unsqueeze(0).unsqueeze(0)  # Add batch and pose dims
    
    # 90 degree rotation matrix
    rot_90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32, device=device)
    rotations_90 = rot_90.unsqueeze(0).unsqueeze(0)  # Shape: (B=1, P=1, 2, 2)
    
    # Back-project at 90°, no shift - this gives us our reference reconstruction
    ref_reconstruction, _ = torch_projectors.backproject_2d_forw(
        projections_fourier, rotations=rotations_90, shifts=None, 
        interpolation=interpolation
    )
    
    # Step 2: Create reference reconstruction at 0° (identity rotation)
    rot_0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=device)
    rotations_0 = rot_0.unsqueeze(0).unsqueeze(0)  # Shape: (B=1, P=1, 2, 2)
    
    ref_reconstruction_0, _ = torch_projectors.backproject_2d_forw(
        projections_fourier, rotations=rotations_0, shifts=None,
        interpolation=interpolation
    )
    
    # Step 3: Initialize zero projections with gradients required
    projections_zero_real = torch.zeros(H, W, requires_grad=True, device=device)
    projections_zero_fourier = torch.fft.rfftn(projections_zero_real, dim=(-2, -1))
    projections_zero_fourier = projections_zero_fourier.unsqueeze(0).unsqueeze(0)  # Add batch and pose dims
    
    # Step 4: Back-project the zero projections at 90°
    reconstruction_zero, _ = torch_projectors.backproject_2d_forw(
        projections_zero_fourier, rotations=rotations_90, shifts=None,
        interpolation=interpolation
    )
    
    # Step 5: Calculate MSE loss and backpropagate (target is ref_reconstruction from step 1)
    loss = complex_mse_loss(reconstruction_zero, ref_reconstruction.detach())
    loss.backward()
    
    # Step 6: Single gradient descent update
    learning_rate = 1.0
    with torch.no_grad():
        projections_zero_real -= learning_rate * projections_zero_real.grad
    
    # Step 7: Calculate normalized cross-correlation with 0° reference
    # (The gradient should make our updated projections look like the 0° back-projection result)
    ref_reconstruction_0_real = torch.fft.irfftn(ref_reconstruction_0[0], dim=(-2, -1), s=(H, W))
    ncc = normalized_cross_correlation(projections_zero_real, ref_reconstruction_0_real)
    
    # Step 8: Create visualizations
    projections_zero_fourier_updated = torch.fft.rfftn(projections_zero_real, dim=(-2, -1))
    ref_reconstruction_0_fourier = ref_reconstruction_0[0]
    
    os.makedirs('test_outputs/2d/back', exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Real space plots
    axes[0, 0].imshow(ref_reconstruction_0_real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 0].set_title('Reference (0° back-proj)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(projections_zero_real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 1].set_title('Updated projections')
    axes[0, 1].axis('off')
    
    # Fourier space plots - real parts
    axes[0, 2].imshow(ref_reconstruction_0_fourier.real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 2].set_title('Reference FFT (Real)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(projections_zero_fourier_updated.real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 3].set_title('Updated FFT (Real)')
    axes[0, 3].axis('off')
    
    # Fourier space plots - imaginary parts
    axes[1, 0].imshow(ref_reconstruction_0_fourier.imag.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[1, 0].set_title('Reference FFT (Imag)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(projections_zero_fourier_updated.imag.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[1, 1].set_title('Updated FFT (Imag)')
    axes[1, 1].axis('off')
    
    # Add text with results
    axes[1, 2].text(0.1, 0.7, f'NCC: {ncc.item():.6f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, f'Loss: {loss.item():.6f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.3, f'Grad norm: {projections_zero_real.grad.norm().item():.6f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'test_outputs/2d/back/gradient_debug_fourier_components_{interpolation}_{device.type}.png')
    plt.close()
    
    print(f"Normalized cross-correlation: {ncc.item():.6f}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient norm: {projections_zero_real.grad.norm().item():.6f}")
    
    # The test - NCC should be reasonable if gradients work correctly
    # Note: Back-projection gradients can be more complex than forward projection
    assert ncc > 0.5, f"Expected NCC > 0.5, got {ncc.item():.6f}"


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
    # Batching tests - multiple projection sets
    ([2.0, 1.0], 3, 1),
    # Batching tests - multiple poses
    ([2.0, 1.0], 1, 4),
])
def test_backproject_shift_gradient_verification_parametrized(device, interpolation, shift_values, batch_size, num_poses):
    """
    Parametrized test to verify shift gradient calculation for back-projection across various scenarios.
    """
    torch.manual_seed(42)
    
    # Test parameters
    H, W = 32, 32
    B = batch_size
    P = num_poses
    
    # Create test shifts - broadcast the same shift to all batches and poses for consistency
    test_shift = torch.tensor(shift_values, dtype=torch.float32, device=device)
    test_shift = test_shift.unsqueeze(0).unsqueeze(0).expand(B, P, 2)  # Shape: (B, P, 2)
    
    # Step 1: Create random projections (one per batch and pose)
    projections_random = torch.randn(B, P, H, H // 2 + 1, dtype=torch.complex64, device=device)
    
    # 15-degree rotation (same for all batches and poses)
    angle_deg = 180.0
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot_15 = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32, device=device)
    rotations = rot_15.unsqueeze(0).unsqueeze(0).expand(B, P, 2, 2)  # Shape: (B, P, 2, 2)
    
    # Step 2: Generate unshifted back-projection at 15 degrees
    reconstruction_unshifted, _ = torch_projectors.backproject_2d_forw(
        projections_random, rotations=rotations, shifts=None,
        interpolation=interpolation
    )
    
    # Step 3: Generate shifted back-projection using our implementation
    shifts_for_our_impl = test_shift.clone().requires_grad_(True)
    reconstruction_shifted_our_impl, _ = torch_projectors.backproject_2d_forw(
        projections_random, rotations=rotations, shifts=shifts_for_our_impl,
        interpolation=interpolation
    )
    
    # Step 4: Manually construct phase modulation for ground truth
    shifts_for_manual = test_shift.clone().requires_grad_(True)
    
    # Create coordinate grids for phase calculation
    ky = torch.arange(H, dtype=torch.float32, device=device)
    ky[ky > H // 2] -= H  # Shift to [-H/2, H/2) range
    kx = torch.arange(H // 2 + 1, dtype=torch.float32, device=device)
    kyy, kxx = torch.meshgrid(ky, kx, indexing='ij')
    
    # Manual phase modulation for all batches and poses
    projections_shifted_manual = torch.zeros_like(projections_random)
    for b in range(B):
        for p in range(P):
            # Manual phase modulation: exp(+2πi(ky*shift_r + kx*shift_c)) for back-projection (conjugate)
            # Note: normalization by H to match the implementation
            phase = 2.0 * math.pi * (kyy * shifts_for_manual[b, p, 0] / H + kxx * shifts_for_manual[b, p, 1] / H)
            phase_factor = torch.complex(torch.cos(phase), torch.sin(phase))
            
            # Apply manual phase modulation to projections before back-projection
            projections_shifted_manual[b, p] = projections_random[b, p] * phase_factor
    
    # Back-project the manually shifted projections
    reconstruction_shifted_manual, _ = torch_projectors.backproject_2d_forw(
        projections_shifted_manual, rotations=rotations, shifts=None,
        interpolation=interpolation
    )
    
    # Step 5: Implement manual complex MSE loss and backprop
    # Calculate losses and gradients
    # Our implementation
    loss_our_impl = complex_mse_loss(reconstruction_shifted_our_impl, reconstruction_unshifted)
    loss_our_impl.backward()
    grad_our_impl = shifts_for_our_impl.grad.clone()
    
    # Manual implementation  
    loss_manual = complex_mse_loss(reconstruction_shifted_manual, reconstruction_unshifted)
    loss_manual.backward()
    grad_manual = shifts_for_manual.grad.clone()
    
    # Step 6: Compare gradients
    grad_diff = torch.abs(grad_our_impl - grad_manual)
    rel_error = grad_diff / (torch.abs(grad_manual) + 1e-8)
    max_abs_error = grad_diff.max().item()
    max_rel_error = rel_error.max().item()
    
    print(f"Back-projection Shift: {shift_values}, Batch: {batch_size}, Poses: {num_poses}")
    print(f"Max absolute error: {max_abs_error:.2e}")
    print(f"Max relative error: {max_rel_error:.2e}")
    
    # Test assertion - gradients should be close
    torch.testing.assert_close(grad_our_impl, grad_manual, atol=1e-4, rtol=1e-2)
    
    # Create visualization
    os.makedirs('test_outputs/2d/back', exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Show first batch for visualization
    # Real parts
    axes[0, 0].imshow(reconstruction_unshifted[0].real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 0].set_title('Unshifted (Real)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(reconstruction_shifted_our_impl[0].real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 1].set_title('Shifted - Our Impl (Real)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(reconstruction_shifted_manual[0].real.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 2].set_title('Shifted - Manual (Real)')
    axes[0, 2].axis('off')
    
    # Imaginary parts
    axes[1, 0].imshow(reconstruction_unshifted[0].imag.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[1, 0].set_title('Unshifted (Imag)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(reconstruction_shifted_our_impl[0].imag.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[1, 1].set_title('Shifted - Our Impl (Imag)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(reconstruction_shifted_manual[0].imag.detach().cpu().numpy(), cmap='viridis', origin='lower')
    axes[1, 2].set_title('Shifted - Manual (Imag)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'test_outputs/2d/back/backproject_shift_gradient_verification_{shift_values[0]}_{shift_values[1]}_B{batch_size}_P{num_poses}_{interpolation}_{device.type}.png')
    plt.close()
    
    print(f"✅ Back-projection shift gradient test passed: {shift_values} (B={batch_size}, P={num_poses})")


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_rotation_gradients_comprehensive(device, interpolation):
    """
    Comprehensive test of rotation gradients for back-projection using finite difference verification.
    This test validates that our analytical rotation gradients match numerical derivatives.
    """

    torch.manual_seed(42)

    print(f"\n🔄 Testing back-projection rotation gradients comprehensively ({interpolation})...")
    
    # Test 1: Finite Difference Verification (most important)
    print("  1️⃣ Finite difference verification...")
    _test_backproject_rotation_finite_difference_accuracy(device, interpolation)
    
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
        projections = torch.randn(1, 1, 16, 9, dtype=torch.complex64, requires_grad=True, device=device)
        
        target_rot = create_rotation_matrix_2d(torch.tensor(target_angle, device=device)).unsqueeze(0).unsqueeze(0)
        target_reconstruction, _ = torch_projectors.backproject_2d_forw(projections.detach(), rotations=target_rot.detach(), interpolation=interpolation)
        
        test_angle = target_angle + test_offset
        test_rot = create_rotation_matrix_2d(torch.tensor(test_angle, device=device)).unsqueeze(0).unsqueeze(0)
        test_rot.requires_grad_(True)
        
        pred_reconstruction, _ = torch_projectors.backproject_2d_forw(projections.detach(), rotations=test_rot, interpolation=interpolation)
        loss = torch.sum((pred_reconstruction - target_reconstruction).abs())
        loss.backward()
        
        # Verify gradients exist and are non-zero
        assert test_rot.grad is not None, f"Scenario {i+1}: No gradients computed"
        grad_norm = test_rot.grad.norm().item()
        assert grad_norm > 1e-8, f"Scenario {i+1}: Gradient norm too small: {grad_norm:.2e}"
        print(f"      Gradient norm: {grad_norm:.6f}")
    
    # Test 3: Optimization Convergence (practical validation)
    print("  3️⃣ Optimization convergence verification...")
    _test_backproject_rotation_optimization_convergence(device, interpolation)
    
    print("✅ All back-projection rotation gradient tests passed!")


def _test_backproject_rotation_finite_difference_accuracy(device, interpolation):
    """Test that analytical gradients match finite differences for back-projection"""

    torch.manual_seed(42)
    
    # Create test data - use smaller, more manageable case
    projections = torch.randn(1, 1, 16, 9, dtype=torch.complex64, requires_grad=True, device=device)
    target_angle = 0.1  # Smaller angle difference
    target_rot = create_rotation_matrix_2d(torch.tensor(target_angle, device=device)).unsqueeze(0).unsqueeze(0)
    target_reconstruction, _ = torch_projectors.backproject_2d_forw(projections.detach(), rotations=target_rot.detach(), interpolation=interpolation)
    
    # Test at slightly different angle
    test_angle = target_angle + 0.05  # Smaller perturbation
    test_rot = create_rotation_matrix_2d(torch.tensor(test_angle, device=device)).unsqueeze(0).unsqueeze(0)
    test_rot.requires_grad_(True)
    
    # Compute analytical gradients
    pred_reconstruction, _ = torch_projectors.backproject_2d_forw(projections.detach(), rotations=test_rot, interpolation=interpolation)
    loss = torch.sum((pred_reconstruction - target_reconstruction).abs())
    loss.backward()
    
    # Compare against finite differences using proper angle perturbation
    eps = 1e-5  # Epsilon for numerical stability
    
    print("    Comparing analytical vs finite difference gradients:")
    
    # Create rotation matrices with perturbed angles
    rot_plus = create_rotation_matrix_2d(torch.tensor(test_angle + eps, device=device)).unsqueeze(0).unsqueeze(0)
    rot_minus = create_rotation_matrix_2d(torch.tensor(test_angle - eps, device=device)).unsqueeze(0).unsqueeze(0)
    
    # Compute finite difference
    reconstruction_plus, _ = torch_projectors.backproject_2d_forw(projections.detach(), rotations=rot_plus, interpolation=interpolation)
    reconstruction_minus, _ = torch_projectors.backproject_2d_forw(projections.detach(), rotations=rot_minus, interpolation=interpolation)
    loss_plus = torch.sum((reconstruction_plus - target_reconstruction).abs())
    loss_minus = torch.sum((reconstruction_minus - target_reconstruction).abs())
    fd_angle_grad = (loss_plus - loss_minus) / (2 * eps)
    
    # Convert matrix gradient to angle gradient using chain rule
    analytical_angle_grad = compute_angle_grad_from_matrix_grad(test_rot.grad[0, 0], torch.tensor(test_angle, device=device))
    abs_error = torch.abs(analytical_angle_grad - fd_angle_grad).item()
    relative_error = abs_error / (torch.abs(fd_angle_grad).item() + 1e-8)
    
    print(f"    Angle gradient: analytical={analytical_angle_grad:.6f}, fd={fd_angle_grad:.6f}, abs_err={abs_error:.2e}, rel_err={relative_error:.2e}")
    
    # Reasonable tolerance - analytical gradients should be close to finite differences
    assert relative_error < 0.01, f"Relative finite difference error {relative_error:.2e} exceeds tolerance"
    print(f"    ✅ Relative finite difference error: {relative_error:.2e}")


def _test_backproject_rotation_optimization_convergence(device, interpolation):
    """Test that optimization converges to correct angle for back-projection"""
    # Create target back-projection at known angle
    torch.manual_seed(42)  # For reproducibility
    projections = torch.randn(1, 1, 16, 9, dtype=torch.complex64, device=device)  # Smaller for stability
    target_angles = [0.2, 0.5, 0.8]  # Positive angles for stability
    
    for target_angle in target_angles:
        print(f"    Optimizing toward angle {target_angle:.3f}...")
        
        target_rot = create_rotation_matrix_2d(torch.tensor(target_angle, device=device)).unsqueeze(0).unsqueeze(0)
        target_reconstruction, _ = torch_projectors.backproject_2d_forw(projections, rotations=target_rot, interpolation=interpolation)
        
        # Initialize optimization close to target (not at 0) for better convergence
        init_angle = target_angle + 0.017  # Start 1 degree (~0.017 radians) away
        learned_angle = torch.tensor(init_angle, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([learned_angle], lr=0.02)  # Lower learning rate for stability
        
        best_loss = float('inf')
        best_angle = init_angle
        
        for step in range(100):  # Fewer steps for speed
            optimizer.zero_grad()
            learned_rot = create_rotation_matrix_2d(learned_angle).unsqueeze(0).unsqueeze(0)
            pred_reconstruction, _ = torch_projectors.backproject_2d_forw(projections, rotations=learned_rot, interpolation=interpolation)
            loss = torch.sum((pred_reconstruction - target_reconstruction).abs().pow(2))  # Manual MSE for complex tensors
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


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_2d_weight_gradients(device, interpolation):
    """
    Tests gradient computation with respect to weights in back-projection.
    """
    torch.manual_seed(42)
    
    B, P, H, W = 1, 2, 16, 16
    W_half = W // 2 + 1
    
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex64, device=device)
    weights = torch.rand(B, P, H, W_half, dtype=torch.float32, device=device, requires_grad=True)
    
    # Identity and 45-degree rotations
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    rotations[0, 0] = torch.eye(2, device=device)
    angle = math.pi / 4  # 45 degrees
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotations[0, 1] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)

    reconstruction, weight_reconstruction = torch_projectors.backproject_2d_forw(
        projections, weights=weights, rotations=rotations, interpolation=interpolation
    )
    
    # Create a loss that depends on both reconstruction and weight reconstruction
    loss = (torch.sum(torch.abs(reconstruction)**2) + 
            0.1 * torch.sum(weight_reconstruction**2))
    loss.backward()
    
    # Check that weight gradients exist and are reasonable
    assert weights.grad is not None
    assert torch.all(torch.isfinite(weights.grad))
    
    grad_norm = weights.grad.norm().item()
    assert grad_norm > 1e-8, f"Weight gradient norm too small: {grad_norm:.2e}"
    
    print(f"Weight gradient norm: {grad_norm:.6f}")
    print(f"Loss: {loss.item():.6f}")
    
    # Test finite difference for weight gradients
    eps = 1e-1
    weights_plus = weights.detach().clone()
    weights_plus[0, 0, 0, 0] += eps
    
    reconstruction_plus, weight_reconstruction_plus = torch_projectors.backproject_2d_forw(
        projections, weights=weights_plus, rotations=rotations, interpolation=interpolation
    )
    loss_plus = (torch.sum(torch.abs(reconstruction_plus)**2) + 
                 0.1 * torch.sum(weight_reconstruction_plus**2))
    
    weights_minus = weights.detach().clone()
    weights_minus[0, 0, 0, 0] -= eps
    
    reconstruction_minus, weight_reconstruction_minus = torch_projectors.backproject_2d_forw(
        projections, weights=weights_minus, rotations=rotations, interpolation=interpolation
    )
    loss_minus = (torch.sum(torch.abs(reconstruction_minus)**2) + 
                  0.1 * torch.sum(weight_reconstruction_minus**2))
    
    fd_grad = (loss_plus - loss_minus) / (2 * eps)
    analytical_grad = weights.grad[0, 0, 0, 0].item()
    
    relative_error = abs(analytical_grad - fd_grad) / (abs(fd_grad) + 1e-8)
    print(f"Weight gradient finite difference check: analytical={analytical_grad:.6f}, fd={fd_grad:.6f}, rel_err={relative_error:.2e}")
    
    assert relative_error < 0.001, f"Weight gradient finite difference error too high: {relative_error:.2e}"