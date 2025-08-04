"""
Basic functionality tests for backproject_2d_forw.

This module tests core back-projection functionality including identity back-projections,
rotations, phase shifts, and batching scenarios. Back-projection is the reverse operation
of forward projection, accumulating projection data into reconstructions.
"""

import torch
import torch_projectors
import pytest
import sys
import os

# Add parent directory to path to import test_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_utils import device, plot_fourier_tensors, plot_real_space_tensors, create_fourier_mask


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_2d_identity(device, interpolation):
    """
    Tests the 2D back-projection with an identity rotation. Multiple projections 
    of the same data should accumulate properly into the reconstruction.
    """
    B, P, H, W = 1, 1, 64, 64
    W_half = W // 2 + 1
    
    # Create projection data (what we're back-projecting)
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex64, device=device)
    rotations = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    reconstruction, weights = torch_projectors.backproject_2d_forw(
        projections,
        rotations=rotations,
        interpolation=interpolation
    )
    assert reconstruction.shape == (B, H, W_half)
    assert weights.shape == (0,)  # Empty when no weights provided

    # For identity rotation, the back-projection should distribute the projection 
    # data according to the interpolation kernel
    plot_fourier_tensors(
        [projections[0, 0].cpu(), reconstruction[0].cpu()],
        ["Projection", "Back-projection"],
        f"test_outputs/2d/back/test_backproject_2d_identity_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])  
def test_backproject_2d_with_weights(device, interpolation):
    """
    Tests back-projection with weight accumulation (e.g., for CTF handling).
    """
    B, P, H, W = 1, 2, 32, 32
    W_half = W // 2 + 1
    
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex64, device=device)
    weights = torch.rand(B, P, H, W_half, dtype=torch.float32, device=device)
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    
    # Identity and 90-degree rotations
    rotations[0, 0] = torch.eye(2, device=device)
    rotations[0, 1] = torch.tensor([[0., -1.], [1., 0.]], device=device)

    reconstruction, weight_reconstruction = torch_projectors.backproject_2d_forw(
        projections,
        weights=weights,
        rotations=rotations,
        interpolation=interpolation
    )
    
    assert reconstruction.shape == (B, H, W_half)
    assert weight_reconstruction.shape == (B, H, W_half)
    
    # Weights should be accumulated and positive
    assert torch.all(weight_reconstruction >= 0)
    
    plot_fourier_tensors(
        [projections[0, 0].cpu(), projections[0, 1].cpu(), 
         reconstruction[0].cpu(), weight_reconstruction[0].cpu()],
        ["Projection 0°", "Projection 90°", "Reconstruction", "Weights"],
        f"test_outputs/2d/back/test_backproject_2d_with_weights_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_2d_rotations(device, interpolation):
    """
    Tests that back-projection correctly handles different rotation angles.
    """
    B, P, H, W = 1, 3, 32, 32
    W_half = W // 2 + 1
    
    # Create a projection with a single peak
    projections = torch.zeros(B, P, H, W_half, dtype=torch.complex64, device=device)
    projections[0, :, 0, 5] = 1.0 + 1.0j  # Same peak in all projections

    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    rotations[0, 0] = torch.eye(2, device=device)
    rotations[0, 1] = torch.tensor([[0., 1.], [-1., 0.]], device=device)  # 90°
    rotations[0, 2] = torch.tensor([[0., -1.], [1., 0.]], device=device)  # -90°

    reconstruction, _ = torch_projectors.backproject_2d_forw(
        projections,
        rotations=rotations,
        interpolation=interpolation
    )

    tensors_to_plot = [projections[0, p].cpu() for p in range(P)] + [reconstruction[0].cpu()]
    titles = [f"Projection {p}" for p in range(P)] + ["Back-projection"]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/2d/back/test_backproject_2d_rotations_{interpolation}_{device.type}.png"
    )
    
    # Verify the reconstruction has accumulated data
    assert torch.sum(torch.abs(reconstruction)) > 0


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_2d_with_phase_shift(device, interpolation):
    """
    Tests back-projection with phase shifts. The shifts should be conjugated
    compared to forward projection.
    """
    B, P, H, W = 1, 1, 32, 32
    W_half = W // 2 + 1
    
    # Create a projection from a real-space dot (similar to forward test)
    proj_real = torch.zeros(H, W, dtype=torch.float32, device=device)
    proj_real[H // 2, W // 2] = 1.0
    projections = torch.fft.rfft2(proj_real).unsqueeze(0).unsqueeze(0)

    rotations = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    shift_r, shift_c = 2.0, 3.0
    shifts = torch.tensor([[[shift_r, shift_c]]], dtype=torch.float32, device=device)

    reconstruction, _ = torch_projectors.backproject_2d_forw(
        projections,
        rotations=rotations,
        shifts=shifts,
        interpolation=interpolation
    )
    
    reconstruction_real = torch.fft.irfft2(reconstruction[0], s=(H, W))

    # For back-projection, the phase shift should be conjugated, which means
    # the real-space result should be shifted in the opposite direction
    expected_real_rolled = torch.roll(proj_real, shifts=(-int(shift_r), -int(shift_c)), dims=(0, 1))
    
    # Apply the same Fourier filtering that back-projection does (low-pass at Nyquist)
    expected_fourier = torch.fft.rfft2(expected_real_rolled)
    
    # Create the same frequency mask as back-projection uses
    radius_cutoff = H / 2.0  # Default is proj_boxsize / 2.0
    mask = create_fourier_mask(expected_fourier.shape, radius_cutoff * radius_cutoff, device)
    expected_fourier[mask] = 0  # Zero out high frequencies
    
    expected_real_filtered = torch.fft.irfft2(expected_fourier, s=(H, W))
    
    plot_real_space_tensors(
        [proj_real.cpu(), reconstruction_real.cpu(), expected_real_filtered.cpu()],
        ["Original projection", "Back-projected", "Expected (filtered)"],
        f"test_outputs/2d/back/test_backproject_2d_with_phase_shift_{interpolation}_{device.type}.png"
    )

    # The back-projection should match the filtered, negatively shifted version
    assert torch.allclose(reconstruction_real, expected_real_filtered, atol=1e-1)


def test_dimension_validation_backproject(device):
    """
    Tests that the validation constraints are properly enforced for back-projection:
    - Projections must be 4D (B, P, height, width/2+1)
    - Output reconstructions are 3D (B, height, width/2+1)
    """
    
    # Test valid back-projection (should pass)
    projections = torch.randn(1, 1, 32, 17, dtype=torch.complex64, device=device)
    rot = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    reconstruction, _ = torch_projectors.backproject_2d_forw(projections, rotations=rot)
    assert reconstruction.shape == (1, 32, 17)
    
    # Test wrong projection dimensions (should fail)
    with pytest.raises(ValueError, match="Projections must be a 4D tensor"):
        bad_projections = torch.randn(1, 32, 17, dtype=torch.complex64, device=device)  # Missing pose dimension
        torch_projectors.backproject_2d_forw(bad_projections, rotations=rot)
        
    # Test weight shape mismatch (should fail)
    with pytest.raises(ValueError, match="Weights shape .* must match projections shape"):
        projections = torch.randn(1, 1, 32, 17, dtype=torch.complex64, device=device)
        bad_weights = torch.randn(1, 1, 32, 16, dtype=torch.float32, device=device)  # Wrong size
        torch_projectors.backproject_2d_forw(projections, weights=bad_weights, rotations=rot)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_2d_accumulation(device, interpolation):
    """
    Tests that multiple projections properly accumulate into the reconstruction.
    """
    B, P, H, W = 1, 4, 16, 16
    W_half = W // 2 + 1
    
    # Create identical projections
    base_projection = torch.randn(H, W_half, dtype=torch.complex64, device=device)
    projections = base_projection.unsqueeze(0).unsqueeze(0).expand(B, P, -1, -1).contiguous()
    
    # All identity rotations
    rotations = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(B, P, -1, -1)

    reconstruction, _ = torch_projectors.backproject_2d_forw(
        projections,
        rotations=rotations,
        interpolation=interpolation
    )
    
    # Single projection back-projection for comparison
    single_reconstruction, _ = torch_projectors.backproject_2d_forw(
        base_projection.unsqueeze(0).unsqueeze(0),
        rotations[:, :1],
        interpolation=interpolation
    )
    
    plot_fourier_tensors(
        [base_projection.cpu(), single_reconstruction[0].cpu(), reconstruction[0].cpu()],
        ["Original projection", "Single back-proj", "4x accumulated"],
        f"test_outputs/2d/back/test_backproject_2d_accumulation_{interpolation}_{device.type}.png"
    )
    
    # The accumulated result should be approximately P times the single result
    # (allowing for some numerical differences due to interpolation)
    ratio = torch.abs(reconstruction[0]) / (torch.abs(single_reconstruction[0]) + 1e-8)
    mean_ratio = torch.mean(ratio[torch.abs(single_reconstruction[0]) > 1e-6]).item()
    assert abs(mean_ratio - P) < 0.5, f"Expected ratio ~{P}, got {mean_ratio}"


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_2d_adjoint_relationship(device, interpolation):
    """
    Tests the adjoint relationship: backproject_2d is the adjoint of project_2d.
    For two tensors A and B: <project_2d(A), B> = <A, backproject_2d(B)>
    """
    B, P, H, W = 1, 2, 16, 16
    W_half = W // 2 + 1
    
    # Create test data
    reconstruction = torch.randn(B, H, W_half, dtype=torch.complex64, device=device)
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex64, device=device)
    
    # 90-degree rotations to eliminate interpolation errors
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    # Identity and 90-degree rotation
    rotations[0, 0] = torch.eye(2, device=device)  # 0°
    rotations[0, 1] = torch.tensor([[0., -1.], [1., 0.]], device=device)  # 90°
    
    # Forward projection: A -> project_2d(A)
    forward_result = torch_projectors.project_2d_forw(
        reconstruction, rotations, output_shape=(H, W), interpolation=interpolation
    )
    
    # Back projection: B -> backproject_2d(B)  
    backward_result, _ = torch_projectors.backproject_2d_forw(
        projections, rotations=rotations, interpolation=interpolation
    )
    
    # Compute inner products: <project_2d(A), B> and <A, backproject_2d(B)>
    inner_product_1 = torch.sum(torch.conj(forward_result) * projections).real
    inner_product_2 = torch.sum(torch.conj(reconstruction) * backward_result).real
    
    plot_fourier_tensors(
        [reconstruction[0].cpu(), forward_result[0, 0].cpu(), 
         projections[0, 0].cpu(), backward_result[0].cpu()],
        ["Original rec", "Forward proj", "Test projections", "Back proj"],
        f"test_outputs/2d/back/test_backproject_2d_adjoint_{interpolation}_{device.type}.png"
    )
    
    print(f"Inner product 1 (<forward, proj>): {inner_product_1.item()}")
    print(f"Inner product 2 (<rec, backward>): {inner_product_2.item()}")
    
    # The adjoint relationship should hold within numerical precision
    relative_error = torch.abs(inner_product_1 - inner_product_2) / (torch.abs(inner_product_1) + 1e-8)
    assert relative_error < 0.01, f"Adjoint relationship failed: relative error {relative_error.item()}"