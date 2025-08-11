"""
Oversampling tests for backproject_2d_forw.

This module tests that oversampling in 2D->2D backprojection produces consistent 
results when compared using normalized cross-correlation to avoid scaling issues.
"""

import torch
import torch_projectors
import pytest
from test_utils import device, plot_real_space_tensors, ifftshift_and_crop, normalized_cross_correlation


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("oversampling_factor", [2.0, 3.0])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_oversampling_consistency_2d(device, batch_size, oversampling_factor, interpolation):
    """
    Test that oversampling in backprojection produces consistent results.
    
    The test:
    1. Creates batched Fourier-space projection noise
    2. Backprojects with oversampling=1 and oversampling=n separately
    3. Converts both results to real space using ifftshift_and_crop with matching oversampling
    4. Computes normalized cross-correlation between results (should be high)
    """
    H, W = 32, 32
    W_half = W // 2 + 1
    
    # Step 1: Create batch of Fourier-space projection noise
    projections = torch.randn(batch_size, 1, H, W_half, dtype=torch.complex64, device=device)
    
    # Set up backprojection parameters - 90 degree rotation
    rotations = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # Random shifts in range -5 to +5
    shifts = (torch.rand(1, 1, 2, dtype=torch.float32, device=device) - 0.5) * 10  # -5 to +5
    
    # Step 2a: Backproject with oversampling=1
    data_rec_1, weight_rec_1 = torch_projectors.backproject_2d_forw(
        projections,
        rotations,
        shifts=shifts,
        interpolation=interpolation,
        oversampling=1.0
    )
    
    # Step 2b: Backproject with oversampling=n
    data_rec_n, weight_rec_n = torch_projectors.backproject_2d_forw(
        projections,
        rotations,
        shifts=shifts,
        interpolation=interpolation,
        oversampling=oversampling_factor
    )
    
    # Step 3: Convert both to real space using ifftshift_and_crop with matching oversampling
    real_rec_1_list = []
    real_rec_n_list = []
    
    for b in range(batch_size):
        # For oversampling=1 result, use oversampling_factor=1
        real_1 = ifftshift_and_crop(torch.fft.irfft2(data_rec_1[b]), 1.0)
        real_rec_1_list.append(real_1)
        
        # For oversampling=n result, use the same oversampling_factor
        real_n = ifftshift_and_crop(torch.fft.irfft2(data_rec_n[b]), oversampling_factor)
        real_rec_n_list.append(real_n)
    
    real_rec_1_batch = torch.stack(real_rec_1_list, dim=0)
    real_rec_n_batch = torch.stack(real_rec_n_list, dim=0)
    
    # Step 4: Compute normalized cross-correlation between results
    correlations = []
    for b in range(batch_size):
        corr = normalized_cross_correlation(real_rec_1_batch[b], real_rec_n_batch[b])
        correlations.append(corr.item())
    
    avg_correlation = sum(correlations) / len(correlations)
    
    # Visualize results for first batch element
    if batch_size > 0:
        plot_real_space_tensors(
            [real_rec_1_batch[0].cpu(),
             real_rec_n_batch[0].cpu(),
             (real_rec_1_batch[0] - real_rec_n_batch[0]).abs().cpu()],
            [f"Oversampling=1.0", f"Oversampling={oversampling_factor}", "Absolute Difference"],
            f"test_outputs/2d/back/test_oversampling_back_b{batch_size}_n{oversampling_factor}_{interpolation}_{device.type}.png"
        )
    
    # Results should be highly correlated (> 0.9)
    # The exact reconstruction should be similar, just with different sampling/resolution
    assert avg_correlation > 0.99, \
        f"Low correlation {avg_correlation:.3f} between oversampling=1 and oversampling={oversampling_factor} " \
        f"for batch_size={batch_size}. Correlations: {correlations}"
    
    print(f"Average correlation: {avg_correlation:.3f} for oversampling factor {oversampling_factor}")


@pytest.mark.parametrize("oversampling_factor", [2.0])
def test_backproject_oversampling_weights_2d(device, oversampling_factor):
    """
    Test backprojection oversampling with weights.
    """
    H, W = 16, 16
    W_half = W // 2 + 1
    batch_size, num_poses = 1, 3
    
    # Create projections and weights
    projections = torch.randn(batch_size, num_poses, H, W_half, dtype=torch.complex64, device=device)
    weights = torch.rand(batch_size, num_poses, H, W_half, dtype=torch.float32, device=device)
    
    # Set up parameters
    rotations = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    rotations = rotations.expand(1, num_poses, -1, -1)
    
    # Backproject with different oversampling
    data_rec_1, weight_rec_1 = torch_projectors.backproject_2d_forw(
        projections, rotations, weights=weights, oversampling=1.0
    )
    
    data_rec_n, weight_rec_n = torch_projectors.backproject_2d_forw(
        projections, rotations, weights=weights, oversampling=oversampling_factor
    )
    
    # Convert to real space and check correlation
    real_1 = ifftshift_and_crop(torch.fft.irfft2(data_rec_1[0]), 1.0)
    real_n = ifftshift_and_crop(torch.fft.irfft2(data_rec_n[0]), oversampling_factor)

    correlation = normalized_cross_correlation(real_1, real_n)
    
    # Should still be highly correlated
    assert correlation > 0.99, f"Low correlation {correlation:.3f} with weights"
    
    # Weight reconstructions should also be consistent
    # Convert real weights to complex (imaginary = 0) before processing
    weight_complex_1 = torch.complex(weight_rec_1[0], torch.zeros_like(weight_rec_1[0]))
    weight_complex_n = torch.complex(weight_rec_n[0], torch.zeros_like(weight_rec_n[0]))
    
    weight_real_1 = ifftshift_and_crop(torch.fft.irfft2(weight_complex_1), 1.0)
    weight_real_n = ifftshift_and_crop(torch.fft.irfft2(weight_complex_n), oversampling_factor)
    
    # Plot the weight reconstructions to see what's happening
    plot_real_space_tensors(
        [weight_rec_1[0].cpu(),  # Raw Fourier-space weights
         weight_rec_n[0].cpu(),  # Raw Fourier-space weights
         weight_real_1.cpu(),    # Real-space weights after processing
         weight_real_n.cpu()],   # Real-space weights after processing
        ["Weight Fourier 1.0", f"Weight Fourier {oversampling_factor}", 
         "Weight Real 1.0", f"Weight Real {oversampling_factor}"],
        f"test_outputs/2d/back/test_oversampling_weights_{oversampling_factor}_{device.type}.png",
        shape=(2, 2)
    )

    weight_correlation = normalized_cross_correlation(weight_real_1, weight_real_n)
    
    # Don't assert on weight correlation for now, just observe
    assert weight_correlation > 0.99, f"Low weight correlation {weight_correlation:.3f}"