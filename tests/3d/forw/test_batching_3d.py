"""
Batching tests for 3D->3D forward projection.

This module tests various batching scenarios including multiple reconstructions,
multiple rotations, and their combinations with broadcasting for 3D->3D projections.
"""

import torch
import torch_projectors
import pytest
import math
from test_utils import device, plot_fourier_tensors


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_reconstructions_single_rotation_3d(device, interpolation):
    """
    Tests that a single set of poses is correctly broadcast to multiple 3D reconstructions.
    """
    B, P, D, H, W = 3, 5, 16, 16, 16
    W_half = W // 2 + 1
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)

    # Create a single set of P random 3D rotation matrices (B_rot=1)
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

    output_shape = (D, H, W)

    projection = torch_projectors.project_3d_forw(
        rec_3d_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over reconstructions and project individually
    expected_projection = torch.zeros_like(projection)
    for b in range(B):
        expected_projection[b] = torch_projectors.project_3d_forw(
            rec_3d_fourier[b].unsqueeze(0), rotations, output_shape=output_shape, interpolation=interpolation
        )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    # Plot central slices
    tensors_to_plot = [rec_3d_fourier[0, D//2].cpu()] + [projection[0, p, D//2].cpu() for p in range(P)]
    titles = ["Original Rec (b=0, z=D/2)"] + [f"Projection (p={p}, z=D/2)" for p in range(P)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/test_batching_multiple_reconstructions_single_rotation_3d_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_single_reconstruction_multiple_rotations_3d(device, interpolation):
    """
    Tests that multiple poses are correctly applied to a single 3D reconstruction.
    """
    B, P, D, H, W = 1, 5, 16, 16, 16
    W_half = W // 2 + 1
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)

    # Create P random 3D rotation matrices for the single reconstruction (B_rot=1)
    angles_z = torch.rand(B, P, device=device) * 2 * math.pi
    rotations = torch.zeros(B, P, 3, 3, dtype=torch.float32, device=device)
    for p in range(P):
        cos_z, sin_z = torch.cos(angles_z[0, p]), torch.sin(angles_z[0, p])
        rotations[0, p] = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)

    output_shape = (D, H, W)

    projection = torch_projectors.project_3d_forw(
        rec_3d_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over poses and project individually
    expected_projection = torch.zeros_like(projection)
    for p in range(P):
        expected_projection[0, p] = torch_projectors.project_3d_forw(
            rec_3d_fourier, rotations[:, p].unsqueeze(1), output_shape=output_shape, interpolation=interpolation
        )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    # Plot central slices
    tensors_to_plot = [rec_3d_fourier[0, D//2].cpu()] + [projection[0, p, D//2].cpu() for p in range(P)]
    titles = ["Original Rec (z=D/2)"] + [f"Projection (p={p}, z=D/2)" for p in range(P)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/test_batching_single_reconstruction_multiple_rotations_3d_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_reconstructions_multiple_rotations_3d(device, interpolation):
    """
    Tests the one-to-one mapping of 3D reconstructions to poses (B_rot=B).
    """
    B, P, D, H, W = 4, 5, 16, 16, 16
    W_half = W // 2 + 1
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)

    # Create BxP random 3D rotation matrices (B_rot=B)
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

    output_shape = (D, H, W)

    projection = torch_projectors.project_3d_forw(
        rec_3d_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over reconstructions and poses and project individually
    expected_projection = torch.zeros_like(projection)
    for b in range(B):
        for p in range(P):
            expected_projection[b, p] = torch_projectors.project_3d_forw(
                rec_3d_fourier[b].unsqueeze(0), rotations[b, p].unsqueeze(0).unsqueeze(0),
                output_shape=output_shape, interpolation=interpolation
            )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    # Plot central slices for first few batches
    tensors_to_plot = []
    titles = []
    for b in range(min(B, 2)):  # Show first 2 batches
        tensors_to_plot.append(rec_3d_fourier[b, D//2].cpu())
        titles.append(f"Original (b={b}, z=D/2)")
        for p in range(P):
            tensors_to_plot.append(projection[b, p, D//2].cpu())
            titles.append(f"Proj (b={b}, p={p}, z=D/2)")

    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/test_batching_multiple_reconstructions_multiple_rotations_3d_{interpolation}_{device.type}.png",
        shape=(min(B, 2), P + 1)
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_shifts_broadcast_3d(device, interpolation):
    """
    Tests that 3D shifts correctly broadcast across batches (B_shift=1).
    """
    B, P, D, H, W = 3, 2, 16, 16, 16
    W_half = W // 2 + 1
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)

    # Identity rotations (B_rot=1)
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(1, P, 1, 1)

    # Single set of 3D shifts broadcast to all batches (B_shift=1)
    shifts = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 1.5, -1.0]
    ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, P, 3]

    projection = torch_projectors.project_3d_forw(
        rec_3d_fourier, rotations, shifts=shifts, interpolation=interpolation
    )

    # Ground truth: each batch should get the same shifts
    for b in range(B):
        expected_b = torch_projectors.project_3d_forw(
            rec_3d_fourier[b].unsqueeze(0), rotations, shifts=shifts, interpolation=interpolation
        )
        assert torch.allclose(projection[b], expected_b, atol=1e-5)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_shifts_per_reconstruction_3d(device, interpolation):
    """
    Tests that per-reconstruction 3D shifts work correctly (B_shift=B).
    """
    B, P, D, H, W = 3, 2, 16, 16, 16
    W_half = W // 2 + 1
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)

    # Identity rotations (B_rot=1)
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(1, P, 1, 1)

    # Different 3D shifts for each batch (B_shift=B)
    shifts = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]],      # Batch 0
        [[2.0, 1.0, 1.0], [-1.0, -2.0, 0.0]],     # Batch 1
        [[0.5, -0.5, 0.5], [3.0, 2.0, -1.5]]      # Batch 2
    ], dtype=torch.float32, device=device)  # [B, P, 3]

    projection = torch_projectors.project_3d_forw(
        rec_3d_fourier, rotations, shifts=shifts, interpolation=interpolation
    )

    # Ground truth: loop over batches with their specific shifts
    expected_projection = torch.zeros_like(projection)
    for b in range(B):
        expected_projection[b] = torch_projectors.project_3d_forw(
            rec_3d_fourier[b].unsqueeze(0), rotations, shifts=shifts[b].unsqueeze(0), interpolation=interpolation
        )

    assert torch.allclose(projection, expected_projection, atol=1e-5)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_combined_broadcasting_3d(device, interpolation):
    """
    Tests combined broadcasting scenarios for 3D->3D projections.
    - B=3 reconstructions
    - B_rot=1 (broadcast rotations)
    - B_shift=3 (per-reconstruction shifts)
    """
    B, P, D, H, W = 3, 4, 16, 16, 16
    W_half = W // 2 + 1
    rec_3d_fourier = torch.randn(B, D, H, W_half, dtype=torch.complex64, device=device)

    # Single set of rotations (B_rot=1)
    angles_z = torch.rand(1, P, device=device) * 2 * math.pi
    rotations = torch.zeros(1, P, 3, 3, dtype=torch.float32, device=device)
    for p in range(P):
        cos_z, sin_z = torch.cos(angles_z[0, p]), torch.sin(angles_z[0, p])
        rotations[0, p] = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)

    # Per-reconstruction 3D shifts (B_shift=B)
    shifts = torch.randn(B, P, 3, dtype=torch.float32, device=device)

    projection = torch_projectors.project_3d_forw(
        rec_3d_fourier, rotations, shifts=shifts, interpolation=interpolation
    )

    # Ground truth: each batch uses broadcast rotation + its own shift
    expected_projection = torch.zeros_like(projection)
    for b in range(B):
        expected_projection[b] = torch_projectors.project_3d_forw(
            rec_3d_fourier[b].unsqueeze(0), rotations, shifts=shifts[b].unsqueeze(0), interpolation=interpolation
        )

    assert torch.allclose(projection, expected_projection, atol=1e-5)
