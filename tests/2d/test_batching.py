"""
Batching tests for forward_project_2d.

This module tests various batching scenarios including multiple reconstructions,
multiple angles, and their combinations.
"""

import torch
import torch_projectors
import pytest
import math
from test_utils import device, plot_fourier_tensors


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_reconstructions_single_angle(device, interpolation):
    """
    Tests that a single set of poses is correctly broadcast to multiple reconstructions.
    """
    B, P, H, W = 3, 5, 16, 16
    W_half = W // 2 + 1
    rec_fourier = torch.randn(B, H, W_half, dtype=torch.complex64, device=device)
    
    # Create a single set of P random rotation matrices
    angles = torch.rand(1, P, device=device) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(1, P, 2, 2, dtype=torch.float32, device=device)
    rotations[0, :, 0, 0] = cos_a
    rotations[0, :, 0, 1] = -sin_a
    rotations[0, :, 1, 0] = sin_a
    rotations[0, :, 1, 1] = cos_a
    
    output_shape = (H, W)

    projection = torch_projectors.forward_project_2d(
        rec_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over reconstructions and project individually
    expected_projection = torch.zeros_like(projection)
    for b in range(B):
        expected_projection[b] = torch_projectors.forward_project_2d(
            rec_fourier[b].unsqueeze(0), rotations, output_shape=output_shape, interpolation=interpolation
        )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    tensors_to_plot = [rec_fourier[0].cpu()] + [projection[0, p].cpu() for p in range(P)]
    titles = ["Original Rec (b=0)"] + [f"Projection (p={p})" for p in range(P)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/test_batching_multiple_reconstructions_single_angle_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_single_reconstruction_multiple_angles(device, interpolation):
    """
    Tests that multiple poses are correctly applied to a single reconstruction.
    """
    B, P, H, W = 1, 5, 16, 16
    W_half = W // 2 + 1
    rec_fourier = torch.randn(B, H, W_half, dtype=torch.complex64, device=device)

    # Create P random rotation matrices for the single reconstruction
    angles = torch.rand(B, P, device=device) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    rotations[:, :, 0, 0] = cos_a
    rotations[:, :, 0, 1] = -sin_a
    rotations[:, :, 1, 0] = sin_a
    rotations[:, :, 1, 1] = cos_a
    
    output_shape = (H, W)

    projection = torch_projectors.forward_project_2d(
        rec_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over poses and project individually
    expected_projection = torch.zeros_like(projection)
    for p in range(P):
        expected_projection[0, p] = torch_projectors.forward_project_2d(
            rec_fourier, rotations[:, p].unsqueeze(1), output_shape=output_shape, interpolation=interpolation
        )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    tensors_to_plot = [rec_fourier[0].cpu()] + [projection[0, p].cpu() for p in range(P)]
    titles = ["Original Rec"] + [f"Projection (p={p})" for p in range(P)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/test_batching_single_reconstruction_multiple_angles_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_reconstructions_multiple_angles(device, interpolation):
    """
    Tests the one-to-one mapping of reconstructions to poses.
    """
    B, P, H, W = 4, 5, 16, 16
    W_half = W // 2 + 1
    rec_fourier = torch.randn(B, H, W_half, dtype=torch.complex64, device=device)

    # Create BxP random rotation matrices
    angles = torch.rand(B, P, device=device) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    rotations[:, :, 0, 0] = cos_a
    rotations[:, :, 0, 1] = -sin_a
    rotations[:, :, 1, 0] = sin_a
    rotations[:, :, 1, 1] = cos_a
    
    output_shape = (H, W)

    projection = torch_projectors.forward_project_2d(
        rec_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over reconstructions and poses and project individually
    expected_projection = torch.zeros_like(projection)
    for b in range(B):
        for p in range(P):
            expected_projection[b, p] = torch_projectors.forward_project_2d(
                rec_fourier[b].unsqueeze(0), rotations[b, p].unsqueeze(0).unsqueeze(0), output_shape=output_shape, interpolation=interpolation
            )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    tensors_to_plot = []
    titles = []
    for b in range(B):
        tensors_to_plot.append(rec_fourier[b].cpu())
        titles.append(f"Original (b={b})")
        for p in range(P):
            tensors_to_plot.append(projection[b, p].cpu())
            titles.append(f"Proj (b={b}, p={p})")
    
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/test_batching_multiple_reconstructions_multiple_angles_{interpolation}_{device.type}.png",
        shape=(B, P + 1)
    )