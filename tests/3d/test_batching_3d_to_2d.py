"""
Batching tests for forward_project_3d_to_2d.

This module tests various batching scenarios including multiple reconstructions,
multiple angles, and their combinations for 3D->2D projections.
"""

import torch
import torch_projectors
import pytest
import math
from test_utils import device, plot_fourier_tensors


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_reconstructions_single_angle_3d_to_2d(device, interpolation):
    """
    Tests that a single set of poses is correctly broadcast to multiple 3D reconstructions.
    """
    B, P, D, H, W = 3, 5, 16, 16, 16  # Cubical volumes
    W_half = W // 2 + 1
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)
    
    # Create a single set of P random 3D rotation matrices
    angles_x = torch.rand(1, P, device=device) * 2 * math.pi
    angles_y = torch.rand(1, P, device=device) * 2 * math.pi
    angles_z = torch.rand(1, P, device=device) * 2 * math.pi
    
    rotations = torch.zeros(1, P, 3, 3, dtype=torch.float32, device=device)
    for p in range(P):
        # Simple rotation around Z axis for this test
        cos_z, sin_z = torch.cos(angles_z[0, p]), torch.sin(angles_z[0, p])
        rotations[0, p] = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
    
    output_shape = (H, W)

    projection = torch_projectors.forward_project_3d_to_2d(
        rec_3d_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over reconstructions and project individually
    expected_projection = torch.zeros_like(projection)
    for b in range(B):
        expected_projection[b] = torch_projectors.forward_project_3d_to_2d(
            rec_3d_fourier[b].unsqueeze(0), rotations, output_shape=output_shape, interpolation=interpolation
        )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    tensors_to_plot = [rec_3d_fourier[0, 0].cpu()] + [projection[0, p].cpu() for p in range(P)]
    titles = ["Original Rec (b=0, z=0)"] + [f"Projection (p={p})" for p in range(P)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/test_batching_multiple_reconstructions_single_angle_3d_to_2d_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_single_reconstruction_multiple_angles_3d_to_2d(device, interpolation):
    """
    Tests that multiple poses are correctly applied to a single 3D reconstruction.
    """
    B, P, D, H, W = 1, 5, 16, 16, 16  # Cubical volumes
    W_half = W // 2 + 1
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)

    # Create P random 3D rotation matrices for the single reconstruction
    angles_z = torch.rand(B, P, device=device) * 2 * math.pi
    rotations = torch.zeros(B, P, 3, 3, dtype=torch.float32, device=device)
    for p in range(P):
        cos_z, sin_z = torch.cos(angles_z[0, p]), torch.sin(angles_z[0, p])
        rotations[0, p] = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
    
    output_shape = (H, W)

    projection = torch_projectors.forward_project_3d_to_2d(
        rec_3d_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over poses and project individually
    expected_projection = torch.zeros_like(projection)
    for p in range(P):
        expected_projection[0, p] = torch_projectors.forward_project_3d_to_2d(
            rec_3d_fourier, rotations[:, p].unsqueeze(1), output_shape=output_shape, interpolation=interpolation
        )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    tensors_to_plot = [rec_3d_fourier[0, 0].cpu()] + [projection[0, p].cpu() for p in range(P)]
    titles = ["Original Rec (z=0)"] + [f"Projection (p={p})" for p in range(P)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/test_batching_single_reconstruction_multiple_angles_3d_to_2d_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_reconstructions_multiple_angles_3d_to_2d(device, interpolation):
    """
    Tests the one-to-one mapping of 3D reconstructions to poses.
    """
    B, P, D, H, W = 4, 5, 16, 16, 16  # Cubical volumes
    W_half = W // 2 + 1
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)

    # Create BxP random 3D rotation matrices
    angles_z = torch.rand(B, P, device=device) * 2 * math.pi
    rotations = torch.zeros(B, P, 3, 3, dtype=torch.float32, device=device)
    for b in range(B):
        for p in range(P):
            cos_z, sin_z = torch.cos(angles_z[b, p]), torch.sin(angles_z[b, p])
            rotations[b, p] = torch.tensor([
                [cos_z, -sin_z, 0],
                [sin_z, cos_z, 0],
                [0, 0, 1]
            ], dtype=torch.float32, device=device)
    
    output_shape = (H, W)

    projection = torch_projectors.forward_project_3d_to_2d(
        rec_3d_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over reconstructions and poses and project individually
    expected_projection = torch.zeros_like(projection)
    for b in range(B):
        for p in range(P):
            expected_projection[b, p] = torch_projectors.forward_project_3d_to_2d(
                rec_3d_fourier[b].unsqueeze(0), rotations[b, p].unsqueeze(0).unsqueeze(0), output_shape=output_shape, interpolation=interpolation
            )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    tensors_to_plot = []
    titles = []
    for b in range(B):
        tensors_to_plot.append(rec_3d_fourier[b, 0].cpu())  # Central slice
        titles.append(f"Original (b={b}, z=0)")
        for p in range(P):
            tensors_to_plot.append(projection[b, p].cpu())
            titles.append(f"Proj (b={b}, p={p})")
    
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/test_batching_multiple_reconstructions_multiple_angles_3d_to_2d_{interpolation}_{device.type}.png",
        shape=(B, P + 1)
    )