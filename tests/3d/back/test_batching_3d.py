"""
Batching tests for 3D->3D back-projection.

This module tests various batching scenarios including multiple projections,
multiple rotations, and their combinations with broadcasting for 3D->3D back-projections.
"""

import torch
import torch_projectors
import pytest
import math
import sys
import os

# Add parent directory to path to import test_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from test_utils import device, plot_fourier_tensors


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_projections_single_rotation_3d(device, interpolation):
    """
    Tests that a single set of poses is correctly broadcast to multiple 3D projections.
    """
    B, P, D, H, W = 3, 5, 16, 16, 16
    W_half = W // 2 + 1
    projections = torch.randn(B, P, D, H, W_half, dtype=torch.complex64, device=device)

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

    reconstruction, _ = torch_projectors.backproject_3d_forw(
        projections, rotations, interpolation=interpolation
    )

    # Ground truth: loop over projection batches and backproject individually
    expected_reconstruction = torch.zeros_like(reconstruction)
    for b in range(B):
        expected_reconstruction[b], _ = torch_projectors.backproject_3d_forw(
            projections[b].unsqueeze(0), rotations, interpolation=interpolation
        )

    assert torch.allclose(reconstruction, expected_reconstruction, atol=1e-5)

    # Plot central slices
    tensors_to_plot = [projections[0, 0, D//2].cpu()] + [reconstruction[b, D//2].cpu() for b in range(B)]
    titles = ["Original Projection (b=0, p=0, z=D/2)"] + [f"Reconstruction (b={b}, z=D/2)" for b in range(B)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/back/test_batching_multiple_projections_single_rotation_3d_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_single_projection_multiple_rotations_3d(device, interpolation):
    """
    Tests that multiple poses are correctly applied to a single 3D projection batch.
    """
    B, P, D, H, W = 1, 5, 16, 16, 16
    W_half = W // 2 + 1
    projections = torch.randn(B, P, D, H, W_half, dtype=torch.complex64, device=device)

    # Create P random 3D rotation matrices for the single batch (B_rot=1)
    angles_z = torch.rand(B, P, device=device) * 2 * math.pi
    rotations = torch.zeros(B, P, 3, 3, dtype=torch.float32, device=device)
    for p in range(P):
        cos_z, sin_z = torch.cos(angles_z[0, p]), torch.sin(angles_z[0, p])
        rotations[0, p] = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)

    reconstruction, _ = torch_projectors.backproject_3d_forw(
        projections, rotations, interpolation=interpolation
    )

    # Ground truth: backproject all poses together
    # (This is more of a sanity check since we can't easily decompose back-projection by pose)
    # We check that the result is consistent and accumulates properly
    reconstruction_subset, _ = torch_projectors.backproject_3d_forw(
        projections[:, :3], rotations[:, :3], interpolation=interpolation
    )

    # The full reconstruction should have more accumulated energy than the subset
    full_energy = torch.sum(torch.abs(reconstruction)**2).item()
    subset_energy = torch.sum(torch.abs(reconstruction_subset)**2).item()
    assert full_energy > subset_energy, "Full reconstruction should have more energy than subset"

    # Plot central slices
    tensors_to_plot = [projections[0, p, D//2].cpu() for p in range(P)] + [reconstruction[0, D//2].cpu()]
    titles = [f"Projection (p={p}, z=D/2)" for p in range(P)] + ["Reconstruction (z=D/2)"]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/back/test_batching_single_projection_multiple_rotations_3d_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_projections_multiple_rotations_3d(device, interpolation):
    """
    Tests the one-to-one mapping of 3D projections to poses (B_rot=B).
    """
    B, P, D, H, W = 4, 5, 16, 16, 16
    W_half = W // 2 + 1
    projections = torch.randn(B, P, D, H, W_half, dtype=torch.complex64, device=device)

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

    reconstruction, _ = torch_projectors.backproject_3d_forw(
        projections, rotations, interpolation=interpolation
    )

    # Ground truth: loop over batches and backproject individually
    expected_reconstruction = torch.zeros_like(reconstruction)
    for b in range(B):
        expected_reconstruction[b], _ = torch_projectors.backproject_3d_forw(
            projections[b].unsqueeze(0), rotations[b].unsqueeze(0),
            interpolation=interpolation
        )

    assert torch.allclose(reconstruction, expected_reconstruction, atol=1e-5)

    # Plot central slices for first few batches
    tensors_to_plot = []
    titles = []
    for b in range(min(B, 2)):  # Show first 2 batches
        tensors_to_plot.append(projections[b, 0, D//2].cpu())
        titles.append(f"Projection (b={b}, p=0, z=D/2)")
        tensors_to_plot.append(reconstruction[b, D//2].cpu())
        titles.append(f"Reconstruction (b={b}, z=D/2)")

    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/back/test_batching_multiple_projections_multiple_rotations_3d_{interpolation}_{device.type}.png",
        shape=(min(B, 2), 2)
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_shifts_broadcast_3d(device, interpolation):
    """
    Tests that 3D shifts correctly broadcast across batches (B_shift=1).
    """
    B, P, D, H, W = 3, 2, 16, 16, 16
    W_half = W // 2 + 1
    projections = torch.randn(B, P, D, H, W_half, dtype=torch.complex64, device=device)

    # Identity rotations (B_rot=1)
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(1, P, 1, 1)

    # Single set of 3D shifts broadcast to all batches (B_shift=1)
    shifts = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 1.5, -1.0]
    ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, P, 3]

    reconstruction, _ = torch_projectors.backproject_3d_forw(
        projections, rotations, shifts=shifts, interpolation=interpolation
    )

    # Ground truth: each batch should get the same shifts
    for b in range(B):
        expected_b, _ = torch_projectors.backproject_3d_forw(
            projections[b].unsqueeze(0), rotations, shifts=shifts, interpolation=interpolation
        )
        assert torch.allclose(reconstruction[b], expected_b, atol=1e-5)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_shifts_per_projection_3d(device, interpolation):
    """
    Tests that per-projection 3D shifts work correctly (B_shift=B).
    """
    B, P, D, H, W = 3, 2, 16, 16, 16
    W_half = W // 2 + 1
    projections = torch.randn(B, P, D, H, W_half, dtype=torch.complex64, device=device)

    # Identity rotations (B_rot=1)
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(1, P, 1, 1)

    # Different 3D shifts for each batch (B_shift=B)
    shifts = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]],      # Batch 0
        [[2.0, 1.0, 1.0], [-1.0, -2.0, 0.0]],     # Batch 1
        [[0.5, -0.5, 0.5], [3.0, 2.0, -1.5]]      # Batch 2
    ], dtype=torch.float32, device=device)  # [B, P, 3]

    reconstruction, _ = torch_projectors.backproject_3d_forw(
        projections, rotations, shifts=shifts, interpolation=interpolation
    )

    # Ground truth: loop over batches with their specific shifts
    expected_reconstruction = torch.zeros_like(reconstruction)
    for b in range(B):
        expected_reconstruction[b], _ = torch_projectors.backproject_3d_forw(
            projections[b].unsqueeze(0), rotations, shifts=shifts[b].unsqueeze(0), interpolation=interpolation
        )

    assert torch.allclose(reconstruction, expected_reconstruction, atol=1e-5)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_combined_broadcasting_3d(device, interpolation):
    """
    Tests combined broadcasting scenarios for 3D->3D back-projections.
    - B=3 projection batches
    - B_rot=1 (broadcast rotations)
    - B_shift=3 (per-batch shifts)
    """
    B, P, D, H, W = 3, 4, 16, 16, 16
    W_half = W // 2 + 1
    projections = torch.randn(B, P, D, H, W_half, dtype=torch.complex64, device=device)

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

    # Per-batch 3D shifts (B_shift=B)
    shifts = torch.randn(B, P, 3, dtype=torch.float32, device=device)

    reconstruction, _ = torch_projectors.backproject_3d_forw(
        projections, rotations, shifts=shifts, interpolation=interpolation
    )

    # Ground truth: each batch uses broadcast rotation + its own shift
    expected_reconstruction = torch.zeros_like(reconstruction)
    for b in range(B):
        expected_reconstruction[b], _ = torch_projectors.backproject_3d_forw(
            projections[b].unsqueeze(0), rotations, shifts=shifts[b].unsqueeze(0), interpolation=interpolation
        )

    assert torch.allclose(reconstruction, expected_reconstruction, atol=1e-5)
