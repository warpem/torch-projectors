"""
Oversampling tests for forward_project_2d.

This module tests that oversampling in projection space produces equivalent 
results to padding and shifting in real space before FFT conversion.
"""

import torch
import torch_projectors
import pytest
from test_utils import device, plot_fourier_tensors, plot_real_space_tensors, pad_and_fftshift, normalized_cross_correlation


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("oversampling_factor", [2.0, 3.0])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_oversampling_equivalence_2d(device, batch_size, oversampling_factor, interpolation):
    """
    Test that oversampling parameter produces equivalent results to 
    manual padding and shifting in real space.
    
    The test:
    1. Creates batched real-space tensors with random noise
    2. Path A: Zero-pad to n times the size, fftshift, rfft, project with oversampling=n
    3. Path B: Use original tensor with pad_and_fftshift(oversampling=1), project with oversampling=1
    4. Compare results - both should be equivalent
    """
    H, W = 32, 32
    
    # Step 1: Create batch of real-space tensors with random noise
    real_tensors = torch.randn(batch_size, H, W, dtype=torch.float32, device=device)
    
    # Path A: Manual padding to n times size, then project with oversampling=n
    fourier_batch_padded = torch.fft.rfft2(pad_and_fftshift(real_tensors, oversampling_factor), norm='forward')
    
    # Path B: Use original tensor processed with oversampling=1
    fourier_batch_original = torch.fft.rfft2(pad_and_fftshift(real_tensors, 1.0), norm='forward')
    
    # Set up projection parameters - 90 degree rotation
    rotations = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    shifts = torch.zeros(1, 1, 2, dtype=torch.float32, device=device)
    
    # Path A: Project padded tensor with oversampling=n
    projection_padded = torch_projectors.project_2d_forw(
        fourier_batch_padded,
        rotations,
        shifts=shifts,
        output_shape=(H, W),
        interpolation=interpolation,
        oversampling=oversampling_factor
    )
    
    # Path B: Project original tensor with oversampling=1  
    projection_original = torch_projectors.project_2d_forw(
        fourier_batch_original,
        rotations,
        shifts=shifts,
        output_shape=(H, W),
        interpolation=interpolation,
        oversampling=1.0
    )
    
    # Visualize results for first batch element
    if batch_size > 0:
        plot_fourier_tensors(
            [fourier_batch_original[0].cpu(),
             fourier_batch_padded[0].cpu(),
             projection_original[0, 0].cpu(),
             projection_padded[0, 0].cpu()],
            ["Original Fourier", "Padded Fourier", "Original Projected", "Padded Projected"],
            f"test_outputs/2d/forw/test_oversampling_2d_b{batch_size}_n{oversampling_factor}_{interpolation}_{device.type}.png",
            shape=(2, 2)
        )
        
        plot_real_space_tensors(
            [pad_and_fftshift(real_tensors[0], 1).cpu(),
             torch.fft.irfft2(projection_original[0, 0].cpu(), s=(H, W)),
             torch.fft.irfft2(projection_padded[0, 0].cpu(), s=(H, W))],
            ["Original Real", "Original Projected", "Padded Projected"],
            f"test_outputs/2d/forw/test_oversampling_2d_real_b{batch_size}_n{oversampling_factor}_{interpolation}_{device.type}.png"
        )

    correlations = []
    for b in range(batch_size):
        corr = normalized_cross_correlation(torch.fft.irfft2(projection_original[b]).cpu(), torch.fft.irfft2(projection_padded[b]).cpu())
        correlations.append(corr.item())
    
    avg_correlation = sum(correlations) / len(correlations)
    
    print(f"Average correlation: {avg_correlation:.3f} for oversampling factor {oversampling_factor}")
    
    # Results should be highly correlated (> 0.99)
    # The exact reconstruction should be similar, just with different intensity scaling
    assert avg_correlation > 0.99, \
        f"Low correlation {avg_correlation:.3f} between oversampling=1 and oversampling={oversampling_factor} " \
        f"for batch_size={batch_size}. Correlations: {correlations}"