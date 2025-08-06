"""
Visual validation tests for torch-projectors back-projection.

This module contains tests that generate visual outputs for manual inspection
and validation of back-projection behavior.
"""

import torch
import torch_projectors
import pytest
import math
import os
import sys

# Add parent directory to path to import test_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_utils import device, plot_fourier_tensors


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_visual_backproject_rotation_validation(device, interpolation):
    """
    Visual test: Create 3 line patterns in Fourier space and back-project them at various rotations.
    This shows how back-projection accumulates data from different orientations.
    """
    os.makedirs('test_outputs/2d/back', exist_ok=True)

    H, W = 64, 64
    W_half = W // 2 + 1
    num_projection_sets = 3
    num_rotations = 5
    line_lengths = [10, 20, 30]
    rotation_increments = [5, 15, 30]

    # Create different projection patterns for each set
    projections = torch.zeros(num_projection_sets, num_rotations, H, W_half, dtype=torch.complex64, device=device)
    for i, length in enumerate(line_lengths):
        # Create a simple line pattern in each projection
        projections[i, :, :length, 0] = 1.0 + 1.0j
        horiz_length = length // 2
        projections[i, :, 0, 1:horiz_length+1] = 1.0 + 1.0j
        projections[i, :, length + 5, 4] = 1.0 + 1.0j
        projections[i, :, 4, horiz_length+5] = 1.0 + 1.0j

    # Create rotation matrices for back-projection
    rotations = torch.zeros(num_projection_sets, num_rotations, 2, 2, dtype=torch.float32, device=device)
    for i, increment in enumerate(rotation_increments):
        for j in range(num_rotations):
            angle_rad = math.radians(increment * (j + 1))
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            rotations[i, j] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)

    # Back-project each set of projections
    reconstructions = []
    for i in range(num_projection_sets):
        reconstruction, _ = torch_projectors.backproject_2d_forw(
            projections[i:i+1], rotations=rotations[i:i+1], interpolation=interpolation
        )
        reconstructions.append(reconstruction[0])

    # Also back-project individual projections to show accumulation effect
    individual_reconstructions = []
    for i in range(num_projection_sets):
        for j in range(num_rotations):
            single_reconstruction, _ = torch_projectors.backproject_2d_forw(
                projections[i:i+1, j:j+1], rotations=rotations[i:i+1, j:j+1], 
                interpolation=interpolation
            )
            individual_reconstructions.append(single_reconstruction[0])

    # Prepare tensors for plotting
    tensors_to_plot = []
    titles = []
    
    # Show accumulated reconstructions and individual contributions
    for i in range(num_projection_sets):
        tensors_to_plot.append(reconstructions[i].cpu())
        titles.append(f'Accumulated (len={line_lengths[i]})')
        
        # Show first few individual reconstructions for this set
        for j in range(min(3, num_rotations)):
            idx = i * num_rotations + j
            tensors_to_plot.append(individual_reconstructions[idx].cpu())
            titles.append(f'{rotation_increments[i] * (j + 1)}° single')
    
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/2d/back/test_visual_backproject_rotation_validation_{interpolation}_{device.type}.png",
        shape=(num_projection_sets, 4)  # 1 accumulated + 3 individual per row
    )

    assert os.path.exists(f'test_outputs/2d/back/test_visual_backproject_rotation_validation_{interpolation}_{device.type}.png')

    # Sanity check: accumulated reconstruction should have higher magnitude than individual ones
    for i in range(num_projection_sets):
        accumulated_magnitude = torch.sum(torch.abs(reconstructions[i])).item()
        
        # Check first individual reconstruction for this set
        single_magnitude = torch.sum(torch.abs(individual_reconstructions[i * num_rotations])).item()
        
        # Accumulated should be larger (but not necessarily num_rotations times due to overlap/interference)
        assert accumulated_magnitude > single_magnitude, f"Accumulated reconstruction should have higher magnitude than single reconstruction"


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_visual_backproject_with_weights_validation(device, interpolation):
    """
    Visual test: Back-project with different weight patterns to show weight accumulation effects.
    """
    os.makedirs('test_outputs/2d/back', exist_ok=True)

    H, W = 32, 32
    W_half = W // 2 + 1
    B, P = 1, 4
    
    # Create projections with a simple pattern
    projections = torch.zeros(B, P, H, W_half, dtype=torch.complex64, device=device)
    projections[0, :, H//4:3*H//4, W_half//4:3*W_half//4] = 1.0 + 0.5j
    
    # Create different weight patterns
    weights = torch.zeros(B, P, H, W_half, dtype=torch.float32, device=device)
    
    # Uniform weights
    weights[0, 0] = 1.0
    
    # Center-weighted (Gaussian-like)
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W_half, device=device), indexing='ij')
    center_y, center_x = H // 2, W_half // 2
    gaussian_weight = torch.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * (H // 6)**2))
    weights[0, 1] = gaussian_weight
    
    # Edge-weighted
    edge_weight = torch.ones(H, W_half, device=device)
    edge_weight[H//4:3*H//4, W_half//4:3*W_half//4] = 0.1
    weights[0, 2] = edge_weight
    
    # Random weights
    weights[0, 3] = torch.rand(H, W_half, device=device)
    
    # Create rotation matrices (different for each projection)
    angles = [0, 45, 90, 135]  # degrees
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32, device=device)
    for p, angle_deg in enumerate(angles):
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotations[0, p] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)

    # Back-project with and without weights
    reconstruction_no_weights, _ = torch_projectors.backproject_2d_forw(
        projections, rotations=rotations, interpolation=interpolation
    )
    
    reconstruction_with_weights, weight_reconstruction = torch_projectors.backproject_2d_forw(
        projections, weights=weights, rotations=rotations, interpolation=interpolation
    )
    
    # Also back-project each projection individually to show weight effects
    individual_reconstructions = []
    individual_weight_reconstructions = []
    
    for p in range(P):
        rec, weight_rec = torch_projectors.backproject_2d_forw(
            projections[:, p:p+1], weights=weights[:, p:p+1], 
            rotations=rotations[:, p:p+1], interpolation=interpolation
        )
        individual_reconstructions.append(rec[0])
        individual_weight_reconstructions.append(weight_rec[0])

    # Prepare for visualization
    tensors_to_plot = [
        # Original projection patterns
        projections[0, 0].cpu(), projections[0, 1].cpu(), projections[0, 2].cpu(), projections[0, 3].cpu(),
        
        # Weight patterns
        weights[0, 0].cpu(), weights[0, 1].cpu(), weights[0, 2].cpu(), weights[0, 3].cpu(),
        
        # Individual weighted reconstructions
        individual_reconstructions[0].cpu(), individual_reconstructions[1].cpu(), 
        individual_reconstructions[2].cpu(), individual_reconstructions[3].cpu(),
        
        # Individual weight reconstructions
        individual_weight_reconstructions[0].cpu(), individual_weight_reconstructions[1].cpu(),
        individual_weight_reconstructions[2].cpu(), individual_weight_reconstructions[3].cpu(),
        
        # Final accumulated results
        reconstruction_no_weights[0].cpu(), reconstruction_with_weights[0].cpu(), weight_reconstruction[0].cpu()
    ]
    
    titles = [
        # Original projections
        'Proj 0°', 'Proj 45°', 'Proj 90°', 'Proj 135°',
        
        # Weight patterns
        'Weights (uniform)', 'Weights (center)', 'Weights (edge)', 'Weights (random)',
        
        # Individual weighted reconstructions
        'Weighted Rec 0°', 'Weighted Rec 45°', 'Weighted Rec 90°', 'Weighted Rec 135°',
        
        # Individual weight reconstructions
        'Weight Rec 0°', 'Weight Rec 45°', 'Weight Rec 90°', 'Weight Rec 135°',
        
        # Final results
        'Final (no weights)', 'Final (with weights)', 'Final weight map'
    ]
    
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/2d/back/test_visual_backproject_with_weights_validation_{interpolation}_{device.type}.png",
        shape=(5, 4)  # 5 rows, 4 columns (last row has 3 items)
    )

    assert os.path.exists(f'test_outputs/2d/back/test_visual_backproject_with_weights_validation_{interpolation}_{device.type}.png')
    
    # Sanity checks
    # 1. Weight reconstruction should be positive
    assert torch.all(weight_reconstruction >= 0), "Weight reconstruction should be non-negative"
    
    # 3. Weight reconstruction should accumulate properly
    expected_weight_sum = torch.sum(weight_reconstruction[0]).item()
    individual_weight_sum = sum(torch.sum(w).item() for w in individual_weight_reconstructions)
    
    # They should be approximately equal (within interpolation error)
    relative_error = abs(expected_weight_sum - individual_weight_sum) / (individual_weight_sum + 1e-8)
    assert relative_error < 0.1, f"Weight accumulation inconsistent: {relative_error:.3f}"


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_visual_backproject_shift_validation(device, interpolation):
    """
    Visual test: Back-project with shifts to show phase shift effects in reconstruction space.
    """
    os.makedirs('test_outputs/2d/back', exist_ok=True)

    H, W = 32, 32
    W_half = W // 2 + 1
    B, P = 1, 4
    
    # Create a projection with a localized pattern
    projections = torch.zeros(B, P, H, W_half, dtype=torch.complex64, device=device)
    
    # Create a "cross" pattern in Fourier space
    projections[0, :, H//2, :] = 1.0 + 0.0j  # Horizontal line
    projections[0, :, :, W_half//2] = 1.0 + 0.0j  # Vertical line (as much as possible in half-space)
    
    # Identity rotations for all projections
    rotations = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(B, P, -1, -1)
    
    # Different shifts for each projection
    shifts = torch.tensor([
        [[0.0, 0.0]],    # No shift
        [[2.0, 0.0]],    # X shift only
        [[0.0, 2.0]],    # Y shift only  
        [[2.0, 2.0]]     # Both X and Y shift
    ], dtype=torch.float32, device=device).transpose(0, 1)  # Shape: (B=1, P=4, 2)

    # Back-project with different shifts
    reconstructions = []
    for p in range(P):
        reconstruction, _ = torch_projectors.backproject_2d_forw(
            projections[:, p:p+1], rotations=rotations[:, p:p+1], 
            shifts=shifts[:, p:p+1], interpolation=interpolation
        )
        reconstructions.append(reconstruction[0])
    
    # Also back-project all together for comparison
    reconstruction_all, _ = torch_projectors.backproject_2d_forw(
        projections, rotations=rotations, shifts=shifts, interpolation=interpolation
    )
    
    # Convert some results to real space for better visualization of shift effects
    real_space_reconstructions = []
    for rec in reconstructions:
        real_space = torch.fft.irfft2(rec, s=(H, W))
        real_space_reconstructions.append(real_space)
    
    real_space_all = torch.fft.irfft2(reconstruction_all[0], s=(H, W))
    
    # Reference: back-project without shifts
    reconstruction_no_shift, _ = torch_projectors.backproject_2d_forw(
        projections, rotations=rotations, interpolation=interpolation
    )
    real_space_no_shift = torch.fft.irfft2(reconstruction_no_shift[0], s=(H, W))
    
    # Prepare for visualization
    tensors_to_plot = [
        # Original projection (same for all)
        projections[0, 0].cpu(),
        
        # Individual Fourier space reconstructions
        reconstructions[0].cpu(), reconstructions[1].cpu(), 
        reconstructions[2].cpu(), reconstructions[3].cpu(),
        
        # Individual real space reconstructions  
        real_space_reconstructions[0].cpu(), real_space_reconstructions[1].cpu(),
        real_space_reconstructions[2].cpu(), real_space_reconstructions[3].cpu(),
        
        # Combined results
        reconstruction_all[0].cpu(), real_space_all.cpu(), 
        reconstruction_no_shift[0].cpu(), real_space_no_shift.cpu()
    ]
    
    titles = [
        # Original
        'Original projection',
        
        # Individual Fourier reconstructions
        'Fourier (no shift)', 'Fourier (X shift)', 'Fourier (Y shift)', 'Fourier (XY shift)',
        
        # Individual real space reconstructions
        'Real (no shift)', 'Real (X shift)', 'Real (Y shift)', 'Real (XY shift)',
        
        # Combined results  
        'Combined Fourier', 'Combined Real', 'No shifts Fourier', 'No shifts Real'
    ]
    
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/2d/back/test_visual_backproject_shift_validation_{interpolation}_{device.type}.png",
        shape=(4, 4)  # 4 rows, 4 columns (last row has 4 items)
    )

    assert os.path.exists(f'test_outputs/2d/back/test_visual_backproject_shift_validation_{interpolation}_{device.type}.png')
    
    # Sanity checks
    # 1. Reconstructions with different shifts should be different
    for i in range(1, P):
        diff = torch.abs(reconstructions[0] - reconstructions[i])
        assert torch.sum(diff) > 1e-6, f"Reconstruction {i} should differ from no-shift case"
    
    # 2. Real space patterns should show the shift effects
    # (This is more of a visual check, but we can verify that they're different)
    for i in range(1, P):
        diff = torch.abs(real_space_reconstructions[0] - real_space_reconstructions[i])
        assert torch.sum(diff) > 1e-6, f"Real space reconstruction {i} should differ from no-shift case"
    
    # 3. Combined result should be different from no-shift result
    diff = torch.abs(reconstruction_all[0] - reconstruction_no_shift[0])
    assert torch.sum(diff) > 1e-6, "Combined shifted result should differ from no-shift result"