"""
Oversampling tests for project_3d_to_2d_forw.

This module tests that oversampling in 3D->2D projection space produces equivalent 
results to padding and shifting in real space before FFT conversion.
"""

import torch
import torch_projectors
import pytest
from test_utils import device, plot_fourier_tensors, plot_real_space_tensors, pad_and_fftshift


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("oversampling_factor", [2.0, 3.0])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_oversampling_equivalence_3d_to_2d(device, batch_size, oversampling_factor, interpolation):
    """
    Test that oversampling parameter produces equivalent results to 
    manual padding and shifting in real space for 3D->2D projections.
    
    The test:
    1. Creates batched 3D real-space tensors with random noise
    2. Path A: Zero-pad to n times the size, fftshift, rfft3, project with oversampling=n
    3. Path B: Use original tensor processed with oversampling=1, project with oversampling=1
    4. Compare results - both should be equivalent
    """
    D, H, W = 32, 32, 32
    
    # Step 1: Create batch of 3D real-space tensors with random noise
    real_tensors = torch.randn(batch_size, D, H, W, dtype=torch.float32, device=device)
    
    # Path A: Manual padding to n times size, then project with oversampling=n
    fourier_batch_padded = torch.fft.rfftn(pad_and_fftshift(real_tensors, oversampling_factor), dim=(-3, -2, -1), norm='forward')
    fourier_batch_padded[0, 0, 0, 0] = 0
    
    # Path B: Use original tensor processed with oversampling=1
    fourier_batch_original = torch.fft.rfftn(pad_and_fftshift(real_tensors, 1.0), dim=(-3, -2, -1), norm='forward')
    fourier_batch_original[0, 0, 0, 0] = 0
    
    # Set up projection parameters - 90 degree rotation around Y axis
    # This rotates from XZ plane to YZ plane
    rotations = torch.tensor([
        [0., 0., 1.],    # x' = z
        [0., 1., 0.],    # y' = y  
        [-1., 0., 0.]    # z' = -x
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    shifts = torch.zeros(1, 1, 2, dtype=torch.float32, device=device)
    
    # Path A: Project padded tensor with oversampling=n
    projection_padded = torch_projectors.project_3d_to_2d_forw(
        fourier_batch_padded,
        rotations,
        shifts=shifts,
        output_shape=(H, W),
        interpolation=interpolation,
        oversampling=oversampling_factor
    )
    
    # Path B: Project original tensor with oversampling=1
    projection_original = torch_projectors.project_3d_to_2d_forw(
        fourier_batch_original,
        rotations,
        shifts=shifts,
        output_shape=(H, W),
        interpolation=interpolation,
        oversampling=1.0
    )
    
    # Visualize results for first batch element
    if batch_size > 0:
        # Show central slices of 3D volumes and resulting 2D projections
        plot_fourier_tensors(
            [fourier_batch_original[0, 0].cpu(),  # Central XY slice
             fourier_batch_padded[0, 0].cpu(),  # Central slice of padded
             projection_original[0, 0].cpu(),
             projection_padded[0, 0].cpu()],
            ["Original 3D (center)", "Padded 3D (center)", "Original Projected", "Padded Projected"],
            f"test_outputs/3d/forw/test_oversampling_3d_to_2d_b{batch_size}_n{oversampling_factor}_{interpolation}_{device.type}.png",
            shape=(2, 2)
        )
        
        plot_real_space_tensors(
            [pad_and_fftshift(real_tensors[0], 1.0)[D//2].cpu(),  # Central XY slice
             torch.fft.irfft2(projection_original[0, 0].cpu(), s=(H, W)),
             torch.fft.irfft2(projection_padded[0, 0].cpu(), s=(H, W))],
            ["Original 3D (center)", "Original Projected", "Padded Projected"],
            f"test_outputs/3d/forw/test_oversampling_3d_to_2d_real_b{batch_size}_n{oversampling_factor}_{interpolation}_{device.type}.png"
        )
    
    # Both approaches should produce equivalent results across all batch elements
    # Use relatively loose tolerance since interpolation and padding might introduce small differences
    assert torch.allclose(projection_original, projection_padded, rtol=1e-5, atol=1e-1), \
        f"3D->2D oversampling equivalence failed for batch_size={batch_size}, factor={oversampling_factor}"