"""
Basic functionality tests for forward_project_3d_to_2d.

This module tests core 3D->2D projection functionality including identity projections,
3D rotations, phase shifts, and batching scenarios using the central slice theorem.
"""

import torch
import torch_projectors
import pytest
import math
import numpy as np
from test_utils import device, plot_fourier_tensors, plot_real_space_tensors, create_fourier_mask


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
def test_forward_project_3d_to_2d_identity(device, interpolation):
    """
    Tests the 3D->2D forward projection with an identity rotation. The output should
    correspond to the central slice (z=0 plane) of the 3D volume, filtered by the
    Fourier radius cutoff.
    """
    B, P, D, H, W = 1, 1, 64, 64, 64
    W_half = W // 2 + 1
    
    # Create a 3D Fourier reconstruction with some structure
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)
    
    # Identity rotation (3x3)
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    output_shape = (H, W)

    projection = torch_projectors.forward_project_3d_to_2d(
        rec_3d_fourier,
        rotations,
        output_shape=output_shape,
        interpolation=interpolation
    )
    assert projection.shape == (B, P, H, W_half)

    # For identity rotation, the projection should sample the central slice (z=0) of the 3D volume
    # Extract central slice manually
    central_slice_index = 0  # z=0 corresponds to index 0 in FFTW format (DC component)
    expected_projection = rec_3d_fourier[:, central_slice_index:central_slice_index+1, :, :].squeeze(1)
    
    # Apply the same Fourier space filtering as the C++ kernel
    radius = min(H / 2.0, (W_half - 1) * 2) 
    radius_cutoff_sq = radius * radius
    
    # Create 2D mask for the projection (not full 3D)
    mask_2d = create_fourier_mask(expected_projection.shape, radius_cutoff_sq, device=device)
    expected_projection[0, mask_2d] = 0

    plot_fourier_tensors(
        [rec_3d_fourier[:, central_slice_index, :, :].cpu(), projection.cpu(), expected_projection.cpu()],
        ["Central Slice", "3D->2D Projection", "Expected (Filtered)"],
        f"test_outputs/3d/test_forward_project_3d_to_2d_identity_{interpolation}_{device.type}"
    )

    # Compare the projection with the filtered central slice
    assert torch.allclose(projection[0, 0], expected_projection[0], atol=1e-5)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_3d_to_2d_rotations(device, interpolation):
    """
    Tests 3D->2D projections with non-identity 3x3 rotations.
    
    This test verifies that 3D rotations produce different projections than the identity case,
    and that the central slice theorem is being applied correctly in 3D.
    """
    B, P, D, H, W = 1, 3, 64, 64, 64
    W_half = W // 2 + 1
    
    # Create a structured 3D volume (e.g., a 3D Gaussian)
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

    projection = torch_projectors.forward_project_3d_to_2d(
        rec_3d_fourier,
        rotations,
        interpolation=interpolation
    )
    
    assert projection.shape == (B, P, H, W_half)
    
    # Check that different rotations produce different projections
    for p1 in range(P):
        for p2 in range(p1 + 1, P):
            # Projections should be significantly different for different rotations
            diff = torch.norm(projection[0, p1] - projection[0, p2])
            assert diff > 1e-3, f"Projections {p1} and {p2} are too similar (diff={diff})"

    plot_fourier_tensors(
        [projection[0, p].cpu() for p in range(P)],
        [f"Rotation {p}" for p in range(P)],
        f"test_outputs/3d/test_forward_project_3d_to_2d_rotations_{interpolation}_{device.type}"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_3d_to_2d_with_phase_shift(device, interpolation):
    """
    Tests 3D->2D projections with 2D translation shifts.
    
    Phase shifts in Fourier space correspond to translations in real space.
    This test verifies that shifts are applied correctly to the 2D projections.
    """
    B, P, D, H, W = 1, 2, 64, 64, 64
    W_half = W // 2 + 1
    
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)
    
    # Identity rotations
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(1, P, 1, 1)
    
    # Different 2D shifts for each projection
    shifts = torch.tensor([
        [0.0, 0.0],      # No shift
        [2.5, -1.8]      # Some shift
    ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, P, 2]

    projection = torch_projectors.forward_project_3d_to_2d(
        rec_3d_fourier,
        rotations,
        shifts=shifts,
        interpolation=interpolation
    )
    
    assert projection.shape == (B, P, H, W_half)
    
    # The projections should be different due to the phase shifts
    diff = torch.norm(projection[0, 0] - projection[0, 1])
    assert diff > 1e-3, f"Shifted and unshifted projections are too similar (diff={diff})"
    
    # Test without shifts for comparison
    projection_no_shift = torch_projectors.forward_project_3d_to_2d(
        rec_3d_fourier,
        rotations[:, :1],  # Just first pose
        interpolation=interpolation
    )
    
    # First projection (no shift) should match the no-shift version
    assert torch.allclose(projection[0, 0], projection_no_shift[0, 0], atol=1e-6)

    plot_fourier_tensors(
        [projection[0, 0].cpu(), projection[0, 1].cpu()],
        ["No Shift", f"Shift ({shifts[0, 1, 0]:.1f}, {shifts[0, 1, 1]:.1f})"],
        f"test_outputs/3d/test_forward_project_3d_to_2d_with_phase_shift_{interpolation}_{device.type}"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_central_slice_theorem_validation(device, interpolation):
    """
    Tests that the central slice theorem is correctly implemented.
    
    For an identity rotation, the 3D->2D projection should exactly match
    the central slice of the 3D volume (after applying the same filtering).
    """
    B, D, H, W = 1, 32, 32, 32
    W_half = W // 2 + 1
    
    # Create a simple 3D structure
    rec_3d_fourier = torch.zeros(B, D, H, W_half, dtype=torch.complex64, device=device)
    
    # Put some signal in the central slice (d=0, which corresponds to the central slice theorem)
    center_d, center_h = 0, H // 2  # Use d=0 for central slice, not D//2
    rec_3d_fourier[0, center_d:center_d+1, center_h-2:center_h+3, 1:4] = torch.complex(
        torch.randn(1, 5, 3, device=device), torch.randn(1, 5, 3, device=device)
    )
    
    # Identity rotation
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # Project with 3D->2D
    projection = torch_projectors.forward_project_3d_to_2d(
        rec_3d_fourier,
        rotations,
        interpolation=interpolation
    )
    
    # Extract central slice manually
    central_slice = rec_3d_fourier[0, center_d, :, :]
    
    # Apply the same filtering that the projection would apply
    radius = min(H / 2.0, (W_half - 1) * 2)
    radius_cutoff_sq = radius * radius
    
    for i in range(H):
        for j in range(W_half):
            coord_r = i if i <= H // 2 else i - H
            coord_c = j
            if coord_r * coord_r + coord_c * coord_c > radius_cutoff_sq:
                central_slice[i, j] = 0
    
    # They should be very close (allowing for minor interpolation differences)
    assert torch.allclose(projection[0, 0], central_slice, atol=1e-4)

    plot_fourier_tensors(
        [central_slice.cpu(), projection[0, 0].cpu(), (projection[0, 0] - central_slice).cpu()],
        ["Central Slice", "3D->2D Projection", "Difference"],
        f"test_outputs/3d/test_central_slice_theorem_validation_{interpolation}_{device.type}"
    )