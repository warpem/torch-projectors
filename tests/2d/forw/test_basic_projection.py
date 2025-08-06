"""
Basic functionality tests for forward_project_2d.

This module tests core projection functionality including identity projections,
rotations, phase shifts, and batching scenarios.
"""

import torch
import torch_projectors
import pytest
import math
from test_utils import device, plot_fourier_tensors, plot_real_space_tensors, create_fourier_mask


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_2d_identity(device, interpolation):
    """
    Tests the 2D forward projection with an identity rotation. The output should
    be a masked version of the input, with values outside the Fourier radius zeroed out.
    """
    B, P, H, W = 1, 1, 6, 6
    W_half = W // 2 + 1
    rec_fourier = torch.randn(B, H, W_half, dtype=torch.complex64, device=device)
    rotations = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    output_shape = (H, W)

    projection = torch_projectors.project_2d_forw(
        rec_fourier,
        rotations,
        output_shape=output_shape,
        interpolation=interpolation
    )
    assert projection.shape == (B, P, H, W_half)

    # Create an expected output by masking the input reconstruction
    expected_projection = rec_fourier.clone()
    
    # Replicate the C++ kernel's masking logic
    radius = min(H / 2.0, (W_half - 1) * 2)
    radius_cutoff_sq = radius * radius
    
    mask = create_fourier_mask(rec_fourier.shape, radius_cutoff_sq, device=device)
    expected_projection[0, mask] = 0

    plot_fourier_tensors(
        [rec_fourier.cpu(), projection.cpu(), expected_projection.cpu()],
        ["Original", "Projection", "Expected (Masked)"],
        f"test_outputs/test_forward_project_2d_identity_{interpolation}_{device.type}.png"
    )

    # Compare the projection with the masked ground truth
    assert torch.allclose(projection[0, 0], expected_projection[0], atol=1e-5)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_2d_rotations(device, interpolation):
    """
    Tests that a single Fourier-space peak is rotated to the correct location.
    """
    B, P, H, W = 1, 3, 32, 32
    W_half = W // 2 + 1
    rec_fourier = torch.zeros(B, H, W_half, dtype=torch.complex64, device=device)
    rec_fourier[0, 0, 5] = 1.0 + 1.0j

    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    rotations[0, 0] = torch.eye(2, device=device)
    rotations[0, 1] = torch.tensor([[0., 1.], [-1., 0.]], device=device)
    rotations[0, 2] = torch.tensor([[0., -1.], [1., 0.]], device=device)
    output_shape = (H, W)

    projection = torch_projectors.project_2d_forw(rec_fourier, rotations, output_shape=output_shape, interpolation=interpolation)

    # Expected peak locations after rotation
    expected_coords = [(0, 5), (5, 0), (H - 5, 0)]

    tensors_to_plot = [rec_fourier.cpu()] + [projection[0, p].cpu() for p in range(P)]
    titles = ["Original"] + [f"Rotation {p}" for p in range(P)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/test_forward_project_2d_rotations_{interpolation}_{device.type}.png"
    )
    
    for p in range(P):
        proj_slice = projection[0, p].abs()
        max_val = proj_slice.max()
        peak_coords = (proj_slice == max_val).nonzero(as_tuple=False)
        
        assert max_val > 0.9
        
        # Check if any of the found peak coordinates match the expected coordinates
        found_match = False
        for coord in peak_coords:
            if coord[0].item() == expected_coords[p][0] and coord[1].item() == expected_coords[p][1]:
                found_match = True
                break
        assert found_match, f"Peak for pose {p} not found at expected location {expected_coords[p]}"


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_2d_with_phase_shift(device, interpolation):
    """
    Tests that a real-space dot, when shifted in Fourier space, lands in the correct
    real-space location after accounting for circular shifts and Fourier masking.
    """
    B, P, H, W = 1, 1, 32, 32
    W_half = W // 2 + 1
    
    rec_real = torch.zeros(H, W, dtype=torch.float32, device=device)
    rec_real[H // 2, W // 2] = 1.0
    rec_fourier = torch.fft.rfft2(rec_real).unsqueeze(0)

    rotations = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    output_shape = (H, W)
    shift_r, shift_c = 2.0, 3.0
    shifts = torch.tensor([[[shift_r, shift_c]]], dtype=torch.float32, device=device)

    projection_fourier = torch_projectors.project_2d_forw(
        rec_fourier, rotations, shifts=shifts, output_shape=output_shape, interpolation=interpolation
    )
    projection_real = torch.fft.irfft2(projection_fourier.squeeze(0), s=(H, W))

    # Create the expected real-space output by rolling and then masking in Fourier space
    expected_real_rolled = torch.roll(rec_real, shifts=(int(shift_r), int(shift_c)), dims=(0, 1))
    expected_fourier = torch.fft.rfft2(expected_real_rolled)
    
    radius = min(H / 2.0, (W_half - 1) * 2)
    radius_cutoff_sq = radius * radius
    mask = create_fourier_mask(expected_fourier.shape, radius_cutoff_sq, device=device)
    expected_fourier[mask] = 0
    
    expected_real = torch.fft.irfft2(expected_fourier, s=(H, W))

    plot_real_space_tensors(
        [rec_real.cpu(), projection_real.cpu(), expected_real.cpu()],
        ["Original", "Shifted (C++)", "Expected (Rolled & Masked)"],
        f"test_outputs/test_forward_project_2d_with_phase_shift_{interpolation}_{device.type}.png"
    )

    # Compare the entire images
    assert torch.allclose(projection_real, expected_real, atol=1e-1)


def test_dimension_validation(device):
    """
    Tests that the new validation constraints are properly enforced:
    - Boxsize must be even
    - Dimensions must be square (boxsize == 2*(boxsize_half-1))
    """
    
    # Test non-square dimensions (should fail)
    with pytest.raises(ValueError, match="expected boxsize .* to match"):
        rec = torch.randn(1, 30, 17, dtype=torch.complex64, device=device)  # 30 != 2*(17-1) = 32
        rot = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        torch_projectors.project_2d_forw(rec, rot)

    # Test odd dimensions (should fail) 
    with pytest.raises(ValueError, match="Boxsize .* must be even"):
        rec = torch.randn(1, 29, 15, dtype=torch.complex64, device=device)  # 29 is odd, should be caught by even check first
        rot = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        torch_projectors.project_2d_forw(rec, rot)
        
    # Test valid square, even dimensions (should pass)
    rec = torch.randn(1, 32, 17, dtype=torch.complex64, device=device)  # 32x32 -> 17 half
    rot = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    proj = torch_projectors.project_2d_forw(rec, rot)
    assert proj.shape == (1, 1, 32, 17)
    
    # Test valid backward projection (should pass)
    proj = torch.randn(1, 1, 32, 17, dtype=torch.complex64, device=device)
    dummy_rec = torch.randn(1, 32, 17, dtype=torch.complex64, device=device)
    rot = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    rec = torch_projectors.project_2d_back(proj, dummy_rec, rot)
    assert rec.shape == (1, 32, 17)