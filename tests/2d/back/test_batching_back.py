"""
Batching tests for backproject_2d_forw.

This module tests various batching scenarios for back-projection including multiple 
projection sets, multiple angles, and their combinations.
"""

import torch
import torch_projectors
import pytest
import math
import sys
import os

# Add parent directory to path to import test_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_utils import device, plot_fourier_tensors


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_projection_sets_single_angle(device, interpolation):
    """
    Tests that a single set of poses is correctly broadcast to multiple projection sets
    when back-projecting.
    """
    B, P, H, W = 3, 5, 16, 16
    W_half = W // 2 + 1
    
    # Multiple projection sets (different for each batch)
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex64, device=device)
    
    # Create a single set of P random rotation matrices (broadcast to all batches)
    angles = torch.rand(1, P, device=device) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(1, P, 2, 2, dtype=torch.float32, device=device)
    rotations[0, :, 0, 0] = cos_a
    rotations[0, :, 0, 1] = -sin_a
    rotations[0, :, 1, 0] = sin_a
    rotations[0, :, 1, 1] = cos_a

    reconstruction, _ = torch_projectors.backproject_2d_forw(
        projections, rotations=rotations, interpolation=interpolation
    )

    # Ground truth: loop over projection sets and back-project individually
    expected_reconstruction = torch.zeros_like(reconstruction)
    for b in range(B):
        expected_reconstruction[b], _ = torch_projectors.backproject_2d_forw(
            projections[b].unsqueeze(0), rotations=rotations, interpolation=interpolation
        )

    assert torch.allclose(reconstruction, expected_reconstruction, atol=1e-5)

    tensors_to_plot = [projections[0, 0].cpu()] + [reconstruction[b].cpu() for b in range(B)]
    titles = ["Sample projection (b=0, p=0)"] + [f"Reconstruction (b={b})" for b in range(B)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/2d/back/test_batching_multiple_projection_sets_single_angle_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_single_projection_set_multiple_angles(device, interpolation):
    """
    Tests that multiple poses are correctly applied to a single projection set
    when back-projecting.
    """
    B, P, H, W = 1, 5, 16, 16
    W_half = W // 2 + 1
    
    # Single projection set
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex64, device=device)

    # Create P random rotation matrices for the single projection set
    angles = torch.rand(B, P, device=device) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    rotations[:, :, 0, 0] = cos_a
    rotations[:, :, 0, 1] = -sin_a
    rotations[:, :, 1, 0] = sin_a
    rotations[:, :, 1, 1] = cos_a

    reconstruction, _ = torch_projectors.backproject_2d_forw(
        projections, rotations=rotations, interpolation=interpolation
    )

    # Ground truth: loop over poses and back-project individually, then sum
    expected_reconstruction = torch.zeros_like(reconstruction)
    for p in range(P):
        single_reconstruction, _ = torch_projectors.backproject_2d_forw(
            projections[:, p:p+1], rotations=rotations[:, p:p+1], interpolation=interpolation
        )
        expected_reconstruction += single_reconstruction

    assert torch.allclose(reconstruction, expected_reconstruction, atol=1e-5)

    tensors_to_plot = [projections[0, p].cpu() for p in range(P)] + [reconstruction[0].cpu()]
    titles = [f"Projection (p={p})" for p in range(P)] + ["Back-projection"]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/2d/back/test_batching_single_projection_set_multiple_angles_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_projection_sets_multiple_angles(device, interpolation):
    """
    Tests the one-to-one mapping of projection sets to poses when back-projecting.
    """
    B, P, H, W = 4, 5, 16, 16
    W_half = W // 2 + 1
    
    # Multiple projection sets
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex64, device=device)

    # Create BxP random rotation matrices
    angles = torch.rand(B, P, device=device) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    rotations[:, :, 0, 0] = cos_a
    rotations[:, :, 0, 1] = -sin_a
    rotations[:, :, 1, 0] = sin_a
    rotations[:, :, 1, 1] = cos_a

    reconstruction, _ = torch_projectors.backproject_2d_forw(
        projections, rotations=rotations, interpolation=interpolation
    )

    # Ground truth: loop over projection sets and poses and back-project individually
    expected_reconstruction = torch.zeros_like(reconstruction)
    for b in range(B):
        for p in range(P):
            single_reconstruction, _ = torch_projectors.backproject_2d_forw(
                projections[b:b+1, p:p+1], rotations=rotations[b:b+1, p:p+1], interpolation=interpolation
            )
            expected_reconstruction[b] += single_reconstruction[0]

    assert torch.allclose(reconstruction, expected_reconstruction, atol=1e-5)

    tensors_to_plot = []
    titles = []
    for b in range(B):
        tensors_to_plot.append(reconstruction[b].cpu())
        titles.append(f"Reconstruction (b={b})")
        # Show first few projections for this batch
        for p in range(min(3, P)):
            tensors_to_plot.append(projections[b, p].cpu())
            titles.append(f"Proj (b={b}, p={p})")
    
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/2d/back/test_batching_multiple_projection_sets_multiple_angles_{interpolation}_{device.type}.png",
        shape=(B, 4)  # reconstruction + 3 projections per batch
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_with_weights(device, interpolation):
    """
    Tests that weight accumulation works correctly with batching.
    """
    B, P, H, W = 2, 3, 16, 16
    W_half = W // 2 + 1
    
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex64, device=device)
    weights = torch.rand(B, P, H, W_half, dtype=torch.float32, device=device)
    
    # Different rotations for each batch and pose
    angles = torch.rand(B, P, device=device) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    rotations[:, :, 0, 0] = cos_a
    rotations[:, :, 0, 1] = -sin_a
    rotations[:, :, 1, 0] = sin_a
    rotations[:, :, 1, 1] = cos_a

    reconstruction, weight_reconstruction = torch_projectors.backproject_2d_forw(
        projections, weights=weights, rotations=rotations, interpolation=interpolation
    )

    # Ground truth: process each batch separately
    expected_reconstruction = torch.zeros_like(reconstruction)
    expected_weight_reconstruction = torch.zeros_like(weight_reconstruction)
    
    for b in range(B):
        rec_b, weight_b = torch_projectors.backproject_2d_forw(
            projections[b:b+1], weights=weights[b:b+1], rotations=rotations[b:b+1], 
            interpolation=interpolation
        )
        expected_reconstruction[b] = rec_b[0]
        expected_weight_reconstruction[b] = weight_b[0]

    assert torch.allclose(reconstruction, expected_reconstruction, atol=1e-5)
    assert torch.allclose(weight_reconstruction, expected_weight_reconstruction, atol=1e-5)
    
    # Verify weights are positive and accumulate properly
    assert torch.all(weight_reconstruction >= 0)

    plot_fourier_tensors(
        [reconstruction[0].cpu(), weight_reconstruction[0].cpu(),
         reconstruction[1].cpu(), weight_reconstruction[1].cpu()],
        ["Reconstruction (b=0)", "Weights (b=0)", "Reconstruction (b=1)", "Weights (b=1)"],
        f"test_outputs/2d/back/test_batching_with_weights_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_broadcast_rotations(device, interpolation):
    """
    Tests broadcasting behavior when rotation batch dimension is 1 but projection batch dimension > 1.
    """
    B, P, H, W = 3, 2, 16, 16
    W_half = W // 2 + 1
    
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex64, device=device)
    
    # Single set of rotations that should broadcast to all projection batches
    angles = torch.rand(1, P, device=device) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(1, P, 2, 2, dtype=torch.float32, device=device)
    rotations[0, :, 0, 0] = cos_a
    rotations[0, :, 0, 1] = -sin_a
    rotations[0, :, 1, 0] = sin_a
    rotations[0, :, 1, 1] = cos_a

    reconstruction, _ = torch_projectors.backproject_2d_forw(
        projections, rotations=rotations, interpolation=interpolation
    )

    # Each batch should use the same rotations
    for b in range(B):
        single_reconstruction, _ = torch_projectors.backproject_2d_forw(
            projections[b:b+1], rotations=rotations, interpolation=interpolation
        )
        assert torch.allclose(reconstruction[b], single_reconstruction[0], atol=1e-5)

    plot_fourier_tensors(
        [reconstruction[b].cpu() for b in range(B)],
        [f"Reconstruction (b={b})" for b in range(B)],
        f"test_outputs/2d/back/test_batching_broadcast_rotations_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_broadcast_shifts(device, interpolation):
    """
    Tests broadcasting behavior for shifts in back-projection.
    """
    B, P, H, W = 2, 3, 16, 16
    W_half = W // 2 + 1
    
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex64, device=device)
    
    # Identity rotations
    rotations = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(B, P, -1, -1)
    
    # Single set of shifts that should broadcast to all projection batches
    shifts = torch.randn(1, P, 2, dtype=torch.float32, device=device) * 2.0

    reconstruction, _ = torch_projectors.backproject_2d_forw(
        projections, rotations=rotations, shifts=shifts, interpolation=interpolation
    )

    # Ground truth: each batch should use the same shifts
    expected_reconstruction = torch.zeros_like(reconstruction)
    for b in range(B):
        expected_reconstruction[b], _ = torch_projectors.backproject_2d_forw(
            projections[b:b+1], rotations=rotations[b:b+1], shifts=shifts, 
            interpolation=interpolation
        )

    assert torch.allclose(reconstruction, expected_reconstruction, atol=1e-5)

    plot_fourier_tensors(
        [reconstruction[b].cpu() for b in range(B)],
        [f"Reconstruction (b={b})" for b in range(B)],
        f"test_outputs/2d/back/test_batching_broadcast_shifts_{interpolation}_{device.type}.png"
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_accumulation_consistency(device, interpolation):
    """
    Tests that back-projection accumulation is consistent across different batching strategies.
    """
    B, P, H, W = 2, 4, 16, 16
    W_half = W // 2 + 1
    
    projections = torch.randn(B, P, H, W_half, dtype=torch.complex64, device=device)
    
    angles = torch.rand(B, P, device=device) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    rotations[:, :, 0, 0] = cos_a
    rotations[:, :, 0, 1] = -sin_a
    rotations[:, :, 1, 0] = sin_a
    rotations[:, :, 1, 1] = cos_a

    # Method 1: Back-project all at once
    reconstruction_batch, _ = torch_projectors.backproject_2d_forw(
        projections, rotations=rotations, interpolation=interpolation
    )

    # Method 2: Back-project each batch separately and verify consistency
    reconstruction_separate = torch.zeros_like(reconstruction_batch)
    for b in range(B):
        reconstruction_separate[b], _ = torch_projectors.backproject_2d_forw(
            projections[b:b+1], rotations=rotations[b:b+1], interpolation=interpolation
        )

    # Method 3: Back-project each projection individually and accumulate
    reconstruction_individual = torch.zeros_like(reconstruction_batch)
    for b in range(B):
        for p in range(P):
            single_rec, _ = torch_projectors.backproject_2d_forw(
                projections[b:b+1, p:p+1], rotations=rotations[b:b+1, p:p+1], 
                interpolation=interpolation
            )
            reconstruction_individual[b] += single_rec[0]

    # All methods should give the same result
    assert torch.allclose(reconstruction_batch, reconstruction_separate, atol=1e-5)
    assert torch.allclose(reconstruction_batch, reconstruction_individual, atol=1e-5)

    plot_fourier_tensors(
        [reconstruction_batch[0].cpu(), reconstruction_separate[0].cpu(), reconstruction_individual[0].cpu(),
         reconstruction_batch[1].cpu(), reconstruction_separate[1].cpu(), reconstruction_individual[1].cpu()],
        ["Batch (b=0)", "Separate (b=0)", "Individual (b=0)",
         "Batch (b=1)", "Separate (b=1)", "Individual (b=1)"],
        f"test_outputs/2d/back/test_batching_accumulation_consistency_{interpolation}_{device.type}.png",
        shape=(2, 3)
    )