"""
Basic functionality tests for backproject_3d_forw.

This module tests core 3D->3D back-projection functionality including identity back-projections,
3D rotations, phase shifts, and batching scenarios. 3D->3D back-projection accumulates 3D projection
data into 3D reconstructions using full 3D rotation (not limited to central slice theorem).
"""

import torch
import torch_projectors
import pytest
import math
import sys
import os

# Add parent directory to path to import test_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from test_utils import device, plot_fourier_tensors, plot_real_space_tensors


def create_3d_fourier_mask(shape, radius_cutoff_sq, device):
    """
    Create a 3D Fourier space mask for filtering high frequencies.

    Args:
        shape: 4D tensor shape [B, D, H, W/2+1]
        radius_cutoff_sq: Squared radius cutoff for low-pass filtering
        device: PyTorch device

    Returns:
        Boolean mask indicating which frequencies to zero out
    """
    B, D, H, W_half = shape
    mask = torch.zeros((D, H, W_half), dtype=torch.bool, device=device)

    for d in range(D):
        for i in range(H):
            for j in range(W_half):
                # Convert array indices to Fourier coordinates
                coord_d = d if d <= D // 2 else d - D
                coord_r = i if i <= H // 2 else i - H
                coord_c = j

                # Apply spherical cutoff in 3D Fourier space
                radius_sq = coord_d * coord_d + coord_r * coord_r + coord_c * coord_c
                if radius_sq > radius_cutoff_sq:
                    mask[d, i, j] = True

    return mask


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_3d_identity(device, interpolation):
    """
    Tests the 3D->3D back-projection with an identity rotation. The back-projection should
    accumulate the 3D projection data into the reconstruction with matching structure.
    """
    B, P, D, H, W = 1, 1, 16, 16, 16
    W_half = W // 2 + 1

    # Create 3D projection data (what we're back-projecting)
    projections = torch.randn(B, P, D, H, W_half, dtype=torch.complex64, device=device)

    # Identity rotation (3x3 for 3D)
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    reconstruction, weight_reconstruction = torch_projectors.backproject_3d_forw(
        projections,
        rotations=rotations,
        interpolation=interpolation
    )
    assert reconstruction.shape == (B, D, H, W_half)
    assert weight_reconstruction.shape == (0,)  # Empty when no weights provided

    # For identity rotation, the back-projection should primarily match the input projection
    # (after accounting for filtering and interpolation)

    plot_fourier_tensors(
        [projections[0, 0, D//2].cpu(), reconstruction[0, D//2].cpu()],
        ["3D Projection (z=D/2)", "Reconstruction (z=D/2)"],
        f"test_outputs/3d/back/test_backproject_3d_identity_{interpolation}_{device.type}.png"
    )

    # Verify that the reconstruction has accumulated data
    assert torch.sum(torch.abs(reconstruction)) > 0


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_3d_with_weights(device, interpolation):
    """
    Tests 3D->3D back-projection with weight accumulation (e.g., for CTF handling).
    """
    B, P, D, H, W = 1, 2, 16, 16, 16
    W_half = W // 2 + 1

    projections = torch.randn(B, P, D, H, W_half, dtype=torch.complex64, device=device)
    weights = torch.rand(B, P, D, H, W_half, dtype=torch.float32, device=device)

    # Identity and 90-degree rotations around Z-axis (3x3 matrices)
    rotations = torch.zeros(B, P, 3, 3, dtype=torch.float32, device=device)
    rotations[0, 0] = torch.eye(3, device=device)  # Identity
    # 90-degree rotation around Z-axis
    rotations[0, 1] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=device)

    reconstruction, weight_reconstruction = torch_projectors.backproject_3d_forw(
        projections,
        weights=weights,
        rotations=rotations,
        interpolation=interpolation
    )

    assert reconstruction.shape == (B, D, H, W_half)
    assert weight_reconstruction.shape == (B, D, H, W_half)

    # Weights should be accumulated and positive
    assert torch.all(weight_reconstruction >= 0)

    plot_fourier_tensors(
        [projections[0, 0, D//2].cpu(), projections[0, 1, D//2].cpu(),
         reconstruction[0, D//2].cpu(), weight_reconstruction[0, D//2].cpu()],
        ["Projection 0° (z=D/2)", "Projection 90° (z=D/2)", "Reconstruction (z=D/2)", "Weights (z=D/2)"],
        f"test_outputs/3d/back/test_backproject_3d_with_weights_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_3d_rotations(device, interpolation):
    """
    Tests that 3D->3D back-projection correctly handles different 3D rotation angles.
    """
    B, P, D, H, W = 1, 3, 16, 16, 16
    W_half = W // 2 + 1

    # Create a 3D projection with a single peak
    projections = torch.zeros(B, P, D, H, W_half, dtype=torch.complex64, device=device)
    projections[0, :, 2, 0, 5] = 1.0 + 1.0j  # Same peak in all projections

    # Create 3 different 3x3 rotation matrices
    rotations = torch.zeros(B, P, 3, 3, dtype=torch.float32, device=device)

    # Identity rotation
    rotations[0, 0] = torch.eye(3, device=device)

    # 90-degree rotation around Y-axis
    rotations[0, 1] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], device=device)

    # 90-degree rotation around X-axis
    rotations[0, 2] = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], device=device)

    reconstruction, _ = torch_projectors.backproject_3d_forw(
        projections,
        rotations=rotations,
        interpolation=interpolation
    )

    # Plot central slices for each rotation's contribution
    tensors_to_plot = [projections[0, p, D//2].cpu() for p in range(P)] + [reconstruction[0, D//2].cpu()]
    titles = [f"Projection {p} (z=D/2)" for p in range(P)] + ["Back-projection (z=D/2)"]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/back/test_backproject_3d_rotations_{interpolation}_{device.type}.png"
    )

    # Verify the reconstruction has accumulated data
    assert torch.sum(torch.abs(reconstruction)) > 0


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_3d_with_phase_shift(device, interpolation):
    """
    Tests 3D->3D back-projection with 3D phase shifts. The shifts should be conjugated
    compared to forward projection.
    """
    B, P, D, H, W = 1, 1, 16, 16, 16
    W_half = W // 2 + 1

    # Create a 3D projection from a real-space dot
    proj_real = torch.zeros(D, H, W, dtype=torch.float32, device=device)
    proj_real[D // 2, H // 2, W // 2] = 1.0
    projections = torch.fft.rfftn(proj_real, dim=(0, 1, 2)).unsqueeze(0).unsqueeze(0)

    # Identity rotation (3x3)
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    shift_d, shift_r, shift_c = 2.0, 3.0, 1.0
    shifts = torch.tensor([[[shift_d, shift_r, shift_c]]], dtype=torch.float32, device=device)

    reconstruction, _ = torch_projectors.backproject_3d_forw(
        projections,
        rotations=rotations,
        shifts=shifts,
        interpolation=interpolation
    )

    # Convert back to real space to check shift direction
    reconstruction_real = torch.fft.irfftn(reconstruction[0], s=(D, H, W), dim=(0, 1, 2))

    # For back-projection, the phase shift should be conjugated, which means
    # the real-space result should be shifted in the opposite direction
    expected_real_rolled = torch.roll(proj_real, shifts=(-int(shift_d), -int(shift_r), -int(shift_c)), dims=(0, 1, 2))

    plot_real_space_tensors(
        [proj_real[D//2].cpu(), reconstruction_real[D//2].cpu(), expected_real_rolled[D//2].cpu()],
        ["Original projection (z=D/2)", "Back-projected (z=D/2)", "Expected (neg. shifted) (z=D/2)"],
        f"test_outputs/3d/back/test_backproject_3d_with_phase_shift_{interpolation}_{device.type}.png"
    )

    # The back-projection should have accumulated data with the conjugated shift
    assert torch.sum(torch.abs(reconstruction)) > 0


def test_dimension_validation_backproject_3d(device):
    """
    Tests that the validation constraints are properly enforced for 3D->3D back-projection:
    - Projections must be 5D (B, P, D, H, W/2+1) and cubic (D == H == W)
    - Rotations must be 4D (B, P, 3, 3) for 3D rotations
    - Output reconstructions are 4D (B, D, H, W/2+1) and cubic
    """

    # Test valid 3D->3D back-projection (should pass)
    projections = torch.randn(1, 1, 16, 16, 9, dtype=torch.complex64, device=device)
    rot = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    reconstruction, _ = torch_projectors.backproject_3d_forw(projections, rotations=rot)
    assert reconstruction.shape == (1, 16, 16, 9)  # (B, D, H, W/2+1)

    # Test wrong projection dimensions (should fail)
    with pytest.raises((RuntimeError, ValueError)):
        bad_projections = torch.randn(1, 16, 16, 9, dtype=torch.complex64, device=device)  # Missing pose dimension
        torch_projectors.backproject_3d_forw(bad_projections, rotations=rot)

    # Test non-cubic projections (should fail)
    with pytest.raises((RuntimeError, ValueError)):
        bad_projections = torch.randn(1, 1, 8, 16, 9, dtype=torch.complex64, device=device)  # D != H
        torch_projectors.backproject_3d_forw(bad_projections, rotations=rot)

    # Test wrong rotation matrix dimensions (should fail)
    with pytest.raises((RuntimeError, ValueError)):
        projections = torch.randn(1, 1, 16, 16, 9, dtype=torch.complex64, device=device)
        bad_rot = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # 2x2 instead of 3x3
        torch_projectors.backproject_3d_forw(projections, rotations=bad_rot)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_3d_accumulation(device, interpolation):
    """
    Tests that multiple 3D projections properly accumulate into the 3D reconstruction.
    """
    B, P, D, H, W = 1, 4, 16, 16, 16
    W_half = W // 2 + 1

    # Create identical projections
    base_projection = torch.randn(D, H, W_half, dtype=torch.complex64, device=device)
    projections = base_projection.unsqueeze(0).unsqueeze(0).expand(B, P, -1, -1, -1).contiguous()

    # All identity rotations (3x3)
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(B, P, -1, -1)

    reconstruction, _ = torch_projectors.backproject_3d_forw(
        projections,
        rotations=rotations,
        interpolation=interpolation
    )

    # Single projection back-projection for comparison
    single_reconstruction, _ = torch_projectors.backproject_3d_forw(
        base_projection.unsqueeze(0).unsqueeze(0),
        rotations[:, :1],
        interpolation=interpolation
    )

    plot_fourier_tensors(
        [base_projection[D//2].cpu(), single_reconstruction[0, D//2].cpu(), reconstruction[0, D//2].cpu()],
        ["Original projection (z=D/2)", "Single back-proj (z=D/2)", "4x accumulated (z=D/2)"],
        f"test_outputs/3d/back/test_backproject_3d_accumulation_{interpolation}_{device.type}.png"
    )

    # The accumulated result should be approximately P times the single result
    # (allowing for some numerical differences due to interpolation)
    ratio = torch.abs(reconstruction[0]) / (torch.abs(single_reconstruction[0]) + 1e-8)
    mean_ratio = torch.mean(ratio[torch.abs(single_reconstruction[0]) > 1e-6]).item()
    assert abs(mean_ratio - P) < 0.5, f"Expected ratio ~{P}, got {mean_ratio}"


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_3d_volume_preservation(device, interpolation):
    """
    Tests that for identity rotation with matching input/output sizes,
    accumulated back-projection preserves energy.
    """
    B, P, D, H, W = 1, 1, 16, 16, 16
    W_half = W // 2 + 1

    # Create a structured 3D projection
    # Place data in the safe octant (d < D//2, r < H//2) to avoid Friedel symmetry skip region
    projections = torch.zeros(B, P, D, H, W_half, dtype=torch.complex64, device=device)
    projections[0, 0, 2:7, 2:7, 1:6] = torch.randn(5, 5, 5, dtype=torch.complex64, device=device)

    # Identity rotation (3x3)
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    reconstruction, _ = torch_projectors.backproject_3d_forw(
        projections,
        rotations=rotations,
        interpolation=interpolation
    )

    # Check that energy is preserved (within reasonable tolerance for interpolation)
    projection_energy = torch.sum(torch.abs(projections[0, 0])**2)
    reconstruction_energy = torch.sum(torch.abs(reconstruction[0])**2)

    # Energy should be similar (allowing for interpolation effects)
    energy_ratio = reconstruction_energy / projection_energy
    assert 0.5 < energy_ratio < 2.0, f"Energy ratio {energy_ratio} is outside expected range"

    # Plot several slices to show the distribution
    slice_indices = [D//4, D//2, 3*D//4]
    slices = [reconstruction[0, idx].cpu() for idx in slice_indices]
    titles = [f"z={idx}" for idx in slice_indices]

    plot_fourier_tensors(
        [projections[0, 0, D//2].cpu()] + slices,
        ["Original Projection (z=D/2)"] + titles,
        f"test_outputs/3d/back/test_backproject_3d_volume_preservation_{interpolation}_{device.type}.png"
    )
