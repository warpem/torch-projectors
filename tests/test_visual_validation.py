"""
Visual validation tests for torch-projectors.

This module contains tests that generate visual outputs for manual inspection
and validation of projection behavior.
"""

import torch
import torch_projectors
import pytest
import math
import os
from test_utils import device, plot_fourier_tensors


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_visual_rotation_validation(device, interpolation):
    """
    Visual test: Create 3 line patterns in Fourier space (vertical + half-length horizontal) and visualize their fftshifted real components after rotations.
    """
    os.makedirs('test_outputs', exist_ok=True)

    H, W = 64, 64
    W_half = W // 2 + 1
    num_reconstructions = 3
    num_rotations = 5
    line_lengths = [10, 20, 30]
    rotation_increments = [5, 15, 30]

    reconstructions = torch.zeros(num_reconstructions, H, W_half, dtype=torch.complex64, device=device)
    for i, length in enumerate(line_lengths):
        reconstructions[i, :length, 0] = 1.0 + 1.0j
        horiz_length = length // 2
        reconstructions[i, 0, 1:horiz_length+1] = 1.0 + 1.0j
        reconstructions[i, length + 5, 4] = 1.0 + 1.0j
        reconstructions[i, 4, horiz_length+5] = 1.0 + 1.0j

    rotations = torch.zeros(num_reconstructions, num_rotations, 2, 2, dtype=torch.float32, device=device)
    for i, increment in enumerate(rotation_increments):
        for j in range(num_rotations):
            angle_rad = math.radians(increment * (j + 1))
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            rotations[i, j] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)

    projections = torch_projectors.forward_project_2d(reconstructions, rotations, output_shape=(H, W), interpolation=interpolation)

    tensors_to_plot = []
    titles = []
    for i in range(num_reconstructions):
        tensors_to_plot.append(reconstructions[i].cpu())
        titles.append(f'Original (len={line_lengths[i]})')
        for j in range(num_rotations):
            tensors_to_plot.append(projections[i, j].cpu())
            titles.append(f'{rotation_increments[i] * (j + 1)}°')
    
    plot_fourier_tensors(
        [t.cpu() for t in tensors_to_plot],
        titles,
        f"test_outputs/test_visual_rotation_validation_{interpolation}_{device.type}.png",
        shape=(num_reconstructions, num_rotations + 1)
    )

    assert os.path.exists(f'test_outputs/test_visual_rotation_validation_{interpolation}_{device.type}.png')

    # Sanity: ensure projections at successive angles differ
    rec_fourier = reconstructions[0].unsqueeze(0)
    prev = None
    for i in range(1, num_rotations + 1):
        angle_deg = rotation_increments[0] * i
        angle_rad = math.radians(angle_deg)
        rot = torch.tensor([[[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]]], dtype=torch.float32, device=device).unsqueeze(0)
        proj = torch_projectors.forward_project_2d(rec_fourier, rot, output_shape=(H, W), interpolation=interpolation)
        if prev is not None:
            assert not torch.allclose(prev, proj[0].real, atol=1e-6)
        prev = proj[0].real