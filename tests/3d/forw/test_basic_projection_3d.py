"""
Basic functionality tests for 3D->3D forward projection.

This module tests core 3D->3D projection functionality including identity projections,
3D rotations, 3D phase shifts, and batching scenarios.
"""

import torch
import torch_projectors
import pytest
import math
import numpy as np
from test_utils import device, plot_fourier_tensors, plot_real_space_tensors


def create_3d_fourier_mask(shape, radius_cutoff_sq, device):
    """
    Create a 3D Fourier space mask for filtering high frequencies.

    Args:
        shape: 5D tensor shape [B, P, D, H, W/2+1]
        radius_cutoff_sq: Squared radius cutoff for low-pass filtering
        device: PyTorch device

    Returns:
        Boolean mask indicating which frequencies to zero out
    """
    B, P, D, H, W_half = shape
    mask = torch.zeros((D, H, W_half), dtype=torch.bool, device=device)

    for i in range(D):
        for j in range(H):
            for k in range(W_half):
                # Convert array indices to Fourier coordinates
                coord_d = i if i <= D // 2 else i - D
                coord_r = j if j <= H // 2 else j - H
                coord_c = k

                # Apply spherical cutoff in 3D Fourier space
                radius_sq = coord_d * coord_d + coord_r * coord_r + coord_c * coord_c
                if radius_sq > radius_cutoff_sq:
                    mask[i, j, k] = True

    return mask


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_3d_identity(device, interpolation):
    """
    Tests the 3D->3D forward projection with an identity rotation. The output should
    match the input volume (after filtering).
    """
    B, P, D, H, W = 1, 1, 16, 16, 16
    W_half = W // 2 + 1

    torch.manual_seed(42)

    # Create a 3D Fourier reconstruction with some structure
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device='cpu').to(device)

    # Identity rotation (3x3)
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    output_shape = (D, H, W)

    projection = torch_projectors.project_3d_forw(
        rec_3d_fourier,
        rotations,
        output_shape=output_shape,
        interpolation=interpolation
    )
    assert projection.shape == (B, P, D, H, W_half)

    # For identity rotation, the projection should match the input (after filtering)
    expected_projection = rec_3d_fourier.clone()

    # Apply the same Fourier space filtering as the C++ kernel
    radius = min(D / 2.0, H / 2.0, (W_half - 1) * 2)
    radius_cutoff_sq = radius * radius

    # Create 3D mask for the projection
    mask_3d = create_3d_fourier_mask(projection.shape, radius_cutoff_sq, device=device)
    expected_projection[0, mask_3d] = 0

    # Plot central slice for visualization
    plot_fourier_tensors(
        [rec_3d_fourier[0, D//2, :, :].cpu(), projection[0, 0, D//2, :, :].cpu(), expected_projection[0, D//2, :, :].cpu()],
        ["Input (z=D/2)", "3D->3D Projection (z=D/2)", "Expected Filtered (z=D/2)"],
        f"test_outputs/3d/test_forward_project_3d_identity_{interpolation}_{device.type}"
    )

    # Compare the projection with the filtered input
    assert torch.allclose(projection[0, 0], expected_projection[0], atol=1e-5)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_3d_rotations(device, interpolation):
    """
    Tests 3D->3D projections with non-identity 3x3 rotations.

    This test verifies that 3D rotations produce different projections than the identity case.
    """
    B, P, D, H, W = 1, 3, 32, 32, 32
    W_half = W // 2 + 1

    # Create a structured 3D volume
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)

    # Create 3 different 3x3 rotation matrices
    angles_x = [0, math.pi/4, math.pi/2]
    angles_y = [0, math.pi/6, math.pi/3]
    angles_z = [0, math.pi/3, math.pi/4]

    rotations = torch.zeros(1, P, 3, 3, dtype=torch.float32, device=device)

    for p in range(P):
        # Create rotation matrices around each axis
        cos_x, sin_x = math.cos(angles_x[p]), math.sin(angles_x[p])
        cos_y, sin_y = math.cos(angles_y[p]), math.sin(angles_y[p])
        cos_z, sin_z = math.cos(angles_z[p]), math.sin(angles_z[p])

        # Rotation around X axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ], dtype=torch.float32, device=device)

        # Rotation around Y axis
        Ry = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=torch.float32, device=device)

        # Rotation around Z axis
        Rz = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)

        # Combined rotation: Rz * Ry * Rx
        rotations[0, p] = Rz @ Ry @ Rx

    projection = torch_projectors.project_3d_forw(
        rec_3d_fourier,
        rotations,
        interpolation=interpolation
    )

    assert projection.shape == (B, P, D, H, W_half)

    # Check that different rotations produce different projections
    for p1 in range(P):
        for p2 in range(p1 + 1, P):
            # Projections should be significantly different for different rotations
            diff_tensor = projection[0, p1] - projection[0, p2]
            diff = torch.norm(diff_tensor.cpu()).to(projection.device)
            assert diff > 1e-3, f"Projections {p1} and {p2} are too similar (diff={diff})"

    # Plot central slices
    plot_fourier_tensors(
        [projection[0, p, D//2].cpu() for p in range(P)],
        [f"Rotation {p} (z=D/2)" for p in range(P)],
        f"test_outputs/3d/test_forward_project_3d_rotations_{interpolation}_{device.type}"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_3d_with_phase_shift(device, interpolation):
    """
    Tests 3D->3D projections with 3D translation shifts.

    Phase shifts in Fourier space correspond to translations in real space.
    This test verifies that 3D shifts are applied correctly.
    """
    B, P, D, H, W = 1, 2, 32, 32, 32
    W_half = W // 2 + 1

    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)

    # Identity rotations
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(1, P, 1, 1)

    # Different 3D shifts for each projection
    shifts = torch.tensor([
        [0.0, 0.0, 0.0],      # No shift
        [2.5, -1.8, 3.2]      # 3D shift
    ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, P, 3]

    projection = torch_projectors.project_3d_forw(
        rec_3d_fourier,
        rotations,
        shifts=shifts,
        interpolation=interpolation
    )

    assert projection.shape == (B, P, D, H, W_half)

    # The projections should be different due to the phase shifts
    diff_tensor = projection[0, 0] - projection[0, 1]
    diff = torch.norm(diff_tensor.cpu()).to(projection.device)
    assert diff > 1e-3, f"Shifted and unshifted projections are too similar (diff={diff})"

    # Test without shifts for comparison
    projection_no_shift = torch_projectors.project_3d_forw(
        rec_3d_fourier,
        rotations[:, :1],  # Just first pose
        interpolation=interpolation
    )

    # First projection (no shift) should match the no-shift version
    assert torch.allclose(projection[0, 0], projection_no_shift[0, 0], atol=1e-6)

    # Plot central slices
    plot_fourier_tensors(
        [projection[0, 0, D//2].cpu(), projection[0, 1, D//2].cpu()],
        ["No Shift (z=D/2)", f"Shift ({shifts[0, 1, 0]:.1f}, {shifts[0, 1, 1]:.1f}, {shifts[0, 1, 2]:.1f}) (z=D/2)"],
        f"test_outputs/3d/test_forward_project_3d_with_phase_shift_{interpolation}_{device.type}"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_3d_output_shape(device, interpolation):
    """
    Tests that 3D->3D projection respects custom output shapes.
    """
    B, D_in, H_in, W_in = 1, 32, 32, 32
    W_in_half = W_in // 2 + 1

    rec_3d_fourier = torch.randn(B, D_in, H_in, W_in_half, dtype=torch.complex64, device=device)

    # Identity rotation
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Test different output shapes
    output_shapes = [
        (16, 16, 16),  # Smaller
        (32, 32, 32),  # Same
        (64, 64, 64),  # Larger
    ]

    for D_out, H_out, W_out in output_shapes:
        W_out_half = W_out // 2 + 1

        projection = torch_projectors.project_3d_forw(
            rec_3d_fourier,
            rotations,
            output_shape=(D_out, H_out, W_out),
            interpolation=interpolation
        )

        expected_shape = (B, 1, D_out, H_out, W_out_half)
        assert projection.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {projection.shape} for output_shape ({D_out}, {H_out}, {W_out})"


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_3d_volume_preservation(device, interpolation):
    """
    Tests that for identity rotation with matching input/output sizes,
    the projection preserves the structure of the input (within filtering bounds).
    """
    B, D, H, W = 1, 16, 16, 16
    W_half = W // 2 + 1

    # Create a simple structured volume with known features
    rec_3d_fourier = torch.zeros(B, D, H, W_half, dtype=torch.complex64, device=device)

    # Add signal in low frequencies
    rec_3d_fourier[0, 2:5, 2:5, 1:4] = torch.complex(
        torch.randn(3, 3, 3, device=device), torch.randn(3, 3, 3, device=device)
    )

    # Identity rotation
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Project with matching size
    projection = torch_projectors.project_3d_forw(
        rec_3d_fourier,
        rotations,
        output_shape=(D, H, W),
        interpolation=interpolation
    )

    # Apply the same filtering to input
    radius = min(D / 2.0, H / 2.0, (W_half - 1) * 2)
    radius_cutoff_sq = radius * radius
    mask_3d = create_3d_fourier_mask(projection.shape, radius_cutoff_sq, device=device)

    expected = rec_3d_fourier.clone()
    expected[0, mask_3d] = 0

    # Should be very close (allowing for minor interpolation differences)
    assert torch.allclose(projection[0, 0], expected[0], atol=1e-4)

    # Plot comparison
    plot_fourier_tensors(
        [rec_3d_fourier[0, D//2].cpu(), projection[0, 0, D//2].cpu(),
         (projection[0, 0, D//2] - expected[0, D//2]).cpu()],
        ["Input (z=D/2)", "Projection (z=D/2)", "Difference (z=D/2)"],
        f"test_outputs/3d/test_forward_project_3d_volume_preservation_{interpolation}_{device.type}"
    )
