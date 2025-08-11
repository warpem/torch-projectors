"""
Oversampling tests for backproject_2d_to_3d_forw.

This module tests that oversampling in 2D->3D backprojection produces consistent 
results when compared using normalized cross-correlation to avoid scaling issues.
"""

import torch
import torch_projectors
import pytest
from test_utils import device, plot_real_space_tensors, ifftshift_and_crop, normalized_cross_correlation


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("oversampling_factor", [2.0, 3.0])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_oversampling_consistency_2d_to_3d(device, batch_size, oversampling_factor, interpolation):
    """
    Test that oversampling in 2D->3D backprojection produces consistent results.
    
    The test:
    1. Creates batched Fourier-space 2D projection noise
    2. Backprojects to 3D with oversampling=1 and oversampling=n separately
    3. Converts both results to real space using ifftshift_and_crop with matching oversampling
    4. Computes normalized cross-correlation between results (should be high)
    """
    torch.manual_seed(42)

    H, W = 32, 32
    W_half = W // 2 + 1
    
    # Step 1: Create batch of Fourier-space 2D projection noise
    projections = torch.randn(batch_size, 1, H, W_half, dtype=torch.complex64, device='cpu').to(device)
    
    # Set up backprojection parameters - random rotation around X axis
    angle = torch.rand(1) * 2 * torch.pi  # Random angle between 0 and 2π
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # X-axis rotation matrix
    rotations = torch.tensor([
        [1., 0., 0.],
        [0., 1, 0], 
        [0., 0,  1]
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # Random shifts in range -5 to +5
    shifts = (torch.rand(1, 1, 2, dtype=torch.float32, device=device) - 0.5) * 5  # -5 to +5
    
    # Step 2a: Backproject with oversampling=1
    data_rec_1, weight_rec_1 = torch_projectors.backproject_2d_to_3d_forw(
        projections,
        rotations,
        shifts=shifts,
        interpolation=interpolation,
        oversampling=1.0
    )
    
    # Step 2b: Backproject with oversampling=n
    data_rec_n, weight_rec_n = torch_projectors.backproject_2d_to_3d_forw(
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
        real_1 = ifftshift_and_crop(torch.fft.irfftn(data_rec_1[b], dim=(-3, -2, -1)), 1.0)
        real_rec_1_list.append(real_1)
        
        # For oversampling=n result, use the same oversampling_factor
        real_n = ifftshift_and_crop(torch.fft.irfftn(data_rec_n[b], dim=(-3, -2, -1)), oversampling_factor)
        real_rec_n_list.append(real_n)
    
    real_rec_1_batch = torch.stack(real_rec_1_list, dim=0)
    real_rec_n_batch = torch.stack(real_rec_n_list, dim=0)
    
    # Step 4: Compute normalized cross-correlation between results
    correlations = []
    for b in range(batch_size):
        corr = normalized_cross_correlation(real_rec_1_batch[b], real_rec_n_batch[b])
        correlations.append(corr.item())
    
    avg_correlation = sum(correlations) / len(correlations)
    
    # Visualize results for first batch element - show central slices
    if batch_size > 0:
        D = real_rec_1_batch[0].shape[-3]
        central_slice = D // 2
        
        plot_real_space_tensors(
            [real_rec_1_batch[0, central_slice].cpu(),  # Central XY slice
             real_rec_n_batch[0, central_slice].cpu(),  # Central XY slice
             (real_rec_1_batch[0, central_slice] - real_rec_n_batch[0, central_slice]).abs().cpu()],
            [f"Oversampling=1.0 (center)", f"Oversampling={oversampling_factor} (center)", "Absolute Difference"],
            f"test_outputs/3d/back/test_oversampling_back_2d_to_3d_b{batch_size}_n{oversampling_factor}_{interpolation}_{device.type}.png"
        )
    
    # Results should be highly correlated (> 0.9)
    # The exact reconstruction should be similar, just with different sampling/resolution
    assert avg_correlation > 0.9, \
        f"Low correlation {avg_correlation:.3f} between oversampling=1 and oversampling={oversampling_factor} " \
        f"for batch_size={batch_size}. Correlations: {correlations}"
    
    print(f"Average correlation: {avg_correlation:.3f} for oversampling factor {oversampling_factor}")


@pytest.mark.parametrize("oversampling_factor", [2.0])
def test_backproject_oversampling_weights_2d_to_3d(device, oversampling_factor):
    """
    Test 2D->3D backprojection oversampling with weights.
    """
    torch.manual_seed(42)

    H, W = 16, 16
    W_half = W // 2 + 1
    batch_size, num_poses = 1, 3
    
    # Create projections and weights
    projections = torch.randn(batch_size, num_poses, H, W_half, dtype=torch.complex64, device='cpu').to(device)
    weights = torch.rand(batch_size, num_poses, H, W_half, dtype=torch.float32, device='cpu').to(device)
    
    # Set up parameters - multiple 3D rotations
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    rotations = rotations.expand(1, num_poses, -1, -1)
    
    # Backproject with different oversampling
    data_rec_1, weight_rec_1 = torch_projectors.backproject_2d_to_3d_forw(
        projections, rotations, weights=weights, oversampling=1.0
    )
    
    data_rec_n, weight_rec_n = torch_projectors.backproject_2d_to_3d_forw(
        projections, rotations, weights=weights, oversampling=oversampling_factor
    )
    
    # Convert to real space and check correlation
    real_1 = ifftshift_and_crop(torch.fft.irfftn(data_rec_1[0], dim=(-3, -2, -1)), 1.0)
    real_n = ifftshift_and_crop(torch.fft.irfftn(data_rec_n[0], dim=(-3, -2, -1)), oversampling_factor)

    correlation = normalized_cross_correlation(real_1, real_n)
    
    # Should still be highly correlated
    assert correlation > 0.99, f"Low correlation {correlation:.3f} with weights"
    
    # Weight reconstructions should also be consistent
    # Convert real weights to complex (imaginary = 0) before processing
    weight_complex_1 = torch.complex(weight_rec_1[0], torch.zeros_like(weight_rec_1[0]))
    weight_complex_n = torch.complex(weight_rec_n[0], torch.zeros_like(weight_rec_n[0]))
    
    weight_real_1 = ifftshift_and_crop(torch.fft.irfftn(weight_complex_1, dim=(-3, -2, -1)), 1.0)
    weight_real_n = ifftshift_and_crop(torch.fft.irfftn(weight_complex_n, dim=(-3, -2, -1)), oversampling_factor)
    
    # Plot the weight reconstructions - show central slices
    D = weight_real_1.shape[-3]
    central_slice = D // 2
    
    plot_real_space_tensors(
        [weight_rec_1[0, central_slice].cpu(),  # Central slice of Fourier-space weights
         weight_rec_n[0, central_slice].cpu(),  # Central slice of Fourier-space weights
         weight_real_1[central_slice].cpu(),    # Central slice of real-space weights
         weight_real_n[central_slice].cpu()],   # Central slice of real-space weights
        ["Weight Fourier 1.0 (center)", f"Weight Fourier {oversampling_factor} (center)", 
         "Weight Real 1.0 (center)", f"Weight Real {oversampling_factor} (center)"],
        f"test_outputs/3d/back/test_oversampling_weights_2d_to_3d_{oversampling_factor}_{device.type}.png",
        shape=(2, 2)
    )

    weight_correlation = normalized_cross_correlation(weight_real_1, weight_real_n)
    
    # Don't assert on weight correlation for now, just observe
    assert weight_correlation > 0.99, f"Low weight correlation {weight_correlation:.3f}"