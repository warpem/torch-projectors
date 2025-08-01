"""
Visual validation tests for 3D->2D projections.

This module contains tests that generate visual outputs for manual inspection
and validation of 3D->2D projection behavior.
"""

import torch
import torch_projectors
import pytest
import math
import os
from test_utils import device, plot_fourier_tensors


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_visual_rotation_validation_3d_to_2d(device, interpolation):
    """
    Visual test: Create 3D structures in Fourier space and visualize their 2D projections 
    at different 3D rotations.
    """
    os.makedirs('test_outputs/3d', exist_ok=True)

    D, H, W = 32, 32, 32  # Cubical volumes
    W_half = W // 2 + 1
    num_reconstructions = 3
    num_rotations = 5

    # Create 3D structures with different characteristics
    reconstructions = torch.zeros(num_reconstructions, D, H, W_half, dtype=torch.complex64, device=device)
    
    # Reconstruction 1
    line_length = 10
    reconstructions[0, :line_length, 0, 0] = 1.0 + 1.0j
    reconstructions[0, 0, :line_length, 0] = 0.5 + 0.5j  # Cross pattern
    reconstructions[0, 0, 0, 1:6] = 0.3 + 0.3j
    
    # Reconstruction 2
    reconstructions[1, :line_length, 0, 0] = 1.0 + 1.0j
    reconstructions[1, 0, :line_length, 0] = 0.5 + 0.5j  # Cross pattern
    reconstructions[1, 0, 0, 1:6] = 0.3 + 0.3j
    
    # Reconstruction 3
    reconstructions[2, :line_length, 0, 0] = 1.0 + 1.0j
    reconstructions[2, 0, :line_length, 0] = 0.5 + 0.5j  # Cross pattern
    reconstructions[2, 0, 0, 1:6] = 0.3 + 0.3j

    # Create rotation matrices - different axes for variety
    rotation_axes = ['x', 'y', 'z']  # One axis per reconstruction
    rotation_increments = [15, 25, 35]  # Different increments
    
    rotations = torch.zeros(num_reconstructions, num_rotations, 3, 3, dtype=torch.float32, device=device)
    
    for i, (axis, increment) in enumerate(zip(rotation_axes, rotation_increments)):
        for j in range(num_rotations):
            angle_rad = math.radians(increment * (j + 1))
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            
            if axis == 'x':
                # Rotation around X axis
                rotations[i, j] = torch.tensor([
                    [1, 0, 0],
                    [0, cos_a, -sin_a],
                    [0, sin_a, cos_a]
                ], dtype=torch.float32, device=device)
            elif axis == 'y':
                # Rotation around Y axis
                rotations[i, j] = torch.tensor([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ], dtype=torch.float32, device=device)
            else:  # axis == 'z'
                # Rotation around Z axis
                rotations[i, j] = torch.tensor([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ], dtype=torch.float32, device=device)

    projections = torch_projectors.forward_project_3d_to_2d(
        reconstructions, rotations, output_shape=(H, W), interpolation=interpolation
    )

    tensors_to_plot = []
    titles = []
    for i in range(num_reconstructions):
        # Show central slice of 3D volume for reference
        tensors_to_plot.append(reconstructions[i, 0].cpu())  # Central slice (z=0)
        titles.append(f'Original 3D (rec={i}, z=0)')
        for j in range(num_rotations):
            tensors_to_plot.append(projections[i, j].cpu())
            angle_deg = rotation_increments[i] * (j + 1)
            titles.append(f'{angle_deg}° ({rotation_axes[i]}-axis)')
    
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/test_visual_rotation_validation_3d_to_2d_{interpolation}_{device.type}.png",
        shape=(num_reconstructions, num_rotations + 1)
    )

    assert os.path.exists(f'test_outputs/3d/test_visual_rotation_validation_3d_to_2d_{interpolation}_{device.type}.png')

    # Sanity: ensure projections at successive angles differ
    rec_3d_fourier = reconstructions[0].unsqueeze(0)
    prev = None
    for i in range(1, num_rotations + 1):
        angle_deg = rotation_increments[0] * i
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        # X-axis rotation for first reconstruction
        rot = torch.tensor([[
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ]], dtype=torch.float32, device=device).unsqueeze(0)
        
        proj = torch_projectors.forward_project_3d_to_2d(
            rec_3d_fourier, rot, output_shape=(H, W), interpolation=interpolation
        )
        if prev is not None:
            assert not torch.allclose(prev, proj[0, 0].real, atol=1e-6)
        prev = proj[0, 0].real


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_visual_shift_validation_3d_to_2d(device, interpolation):
    """
    Visual test: Create 3D structures and visualize how 2D shifts affect the projections.
    """
    os.makedirs('test_outputs/3d', exist_ok=True)

    D, H, W = 24, 24, 24  # Cubical volumes
    W_half = W // 2 + 1
    num_reconstructions = 2
    num_shifts = 4

    # Create 3D structures
    reconstructions = torch.zeros(num_reconstructions, D, H, W_half, dtype=torch.complex64, device=device)
    
    # Reconstruction 1: Asymmetric structure to show shift effects clearly
    reconstructions[0, 0, 0, 0] = 2.0 + 2.0j  # Origin
    reconstructions[0, 2, 3, 1] = 1.5 + 1.5j  # Off-center point
    reconstructions[0, :5, 0, 0] = 1.0 + 1.0j  # Line along Z
    reconstructions[0, 0, :5, 2] = 0.8 + 0.8j  # Line along Y
    
    # Reconstruction 2: Asymmetric pattern to ensure shift effects are visible
    reconstructions[1, 0, 0, 0] = 1.5 + 1.5j  # Origin
    reconstructions[1, 1, 2, 1] = 1.0 + 1.0j  # Off-center point  
    reconstructions[1, 0, 0:4, 0] = 0.8 + 0.8j  # Line along Y
    reconstructions[1, 0:3, 1, 2] = 0.6 + 0.6j  # Line along Z at (Y=1, X=2)

    # Identity rotations for both reconstructions - need to match number of shifts
    rotations = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(num_reconstructions, num_shifts, 3, 3)

    # Different 2D shifts
    shift_values = [
        [0.0, 0.0],    # No shift
        [2.0, 0.0],    # X shift only
        [0.0, 2.0],    # Y shift only  
        [1.5, -1.5]    # Both X and Y shift
    ]
    
    shifts = torch.tensor(shift_values, dtype=torch.float32, device=device).unsqueeze(0).expand(num_reconstructions, num_shifts, 2)

    projections = torch_projectors.forward_project_3d_to_2d(
        reconstructions, rotations, shifts=shifts, output_shape=(H, W), interpolation=interpolation
    )

    tensors_to_plot = []
    titles = []
    for i in range(num_reconstructions):
        # Show central slice of 3D volume for reference
        tensors_to_plot.append(reconstructions[i, 0].cpu())  # Central slice (z=0)
        titles.append(f'Original 3D (rec={i}, z=0)')
        for j in range(num_shifts):
            tensors_to_plot.append(projections[i, j].cpu())
            shift_x, shift_y = shift_values[j]
            titles.append(f'Shift ({shift_x:.1f}, {shift_y:.1f})')
    
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/test_visual_shift_validation_3d_to_2d_{interpolation}_{device.type}.png",
        shape=(num_reconstructions, num_shifts + 1)
    )

    assert os.path.exists(f'test_outputs/3d/test_visual_shift_validation_3d_to_2d_{interpolation}_{device.type}.png')

    # Sanity: ensure shifted projections differ from unshifted
    for i in range(num_reconstructions):
        unshifted = projections[i, 0]  # No shift
        for j in range(1, num_shifts):
            shifted = projections[i, j]
            diff = torch.norm(unshifted - shifted)
            assert diff > 1e-4, f"Shifted projection {j} should differ from unshifted (diff={diff:.2e})"


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_visual_central_slice_theorem_3d_to_2d(device, interpolation):
    """
    Visual validation of the central slice theorem for 3D->2D projections.
    Shows that identity rotation gives the central slice of the 3D volume.
    """
    os.makedirs('test_outputs/3d', exist_ok=True)

    D, H, W = 16, 16, 16  # Smaller cubical volumes for clearer visualization
    W_half = W // 2 + 1

    # Create a 3D structure with clear features in different slices
    rec_3d_fourier = torch.zeros(1, D, H, W_half, dtype=torch.complex64, device=device)
    
    # Put different patterns in different Z slices
    # Central slice (z=0): Strong signal
    rec_3d_fourier[0, 0, 4:8, 1:4] = 2.0 + 2.0j
    rec_3d_fourier[0, 0, 2, 2] = 3.0 + 3.0j
    
    # Other slices: Different patterns
    rec_3d_fourier[0, 2, 6:10, 2:5] = 1.0 + 1.0j  # Slice z=2
    rec_3d_fourier[0, 4, 1:3, 1:3] = 1.5 + 1.5j   # Slice z=4
    
    # Identity rotation - should give central slice
    identity_rot = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # Small rotation around Y axis - should give different result
    angle_rad = math.radians(10)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    small_rot_y = torch.tensor([[
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ]], dtype=torch.float32, device=device).unsqueeze(0)
    
    # Small rotation around X axis  
    small_rot_x = torch.tensor([[
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ]], dtype=torch.float32, device=device).unsqueeze(0)

    # Generate projections
    proj_identity = torch_projectors.forward_project_3d_to_2d(
        rec_3d_fourier, identity_rot, output_shape=(H, W), interpolation=interpolation
    )
    proj_rot_y = torch_projectors.forward_project_3d_to_2d(
        rec_3d_fourier, small_rot_y, output_shape=(H, W), interpolation=interpolation
    )
    proj_rot_x = torch_projectors.forward_project_3d_to_2d(
        rec_3d_fourier, small_rot_x, output_shape=(H, W), interpolation=interpolation
    )
    
    # Extract actual central slice for comparison
    central_slice = rec_3d_fourier[0, 0]  # z=0 slice
    
    # Create visualizations
    tensors_to_plot = [
        central_slice.cpu(),
        rec_3d_fourier[0, 2].cpu(),  # z=2 slice
        rec_3d_fourier[0, 4].cpu(),  # z=4 slice
        proj_identity[0, 0].cpu(),
        proj_rot_y[0, 0].cpu(),
        proj_rot_x[0, 0].cpu()
    ]
    
    titles = [
        'Central Slice (z=0)',
        'Z Slice (z=2)', 
        'Z Slice (z=4)',
        'Identity Projection',
        'Y-Rotation (10°)',
        'X-Rotation (10°)'
    ]
    
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/3d/test_visual_central_slice_theorem_3d_to_2d_{interpolation}_{device.type}.png",
        shape=(2, 3)
    )

    assert os.path.exists(f'test_outputs/3d/test_visual_central_slice_theorem_3d_to_2d_{interpolation}_{device.type}.png')
    
    # Verify that identity projection is closest to central slice
    diff_identity = torch.norm(proj_identity[0, 0] - central_slice)
    diff_rot_y = torch.norm(proj_rot_y[0, 0] - central_slice) 
    diff_rot_x = torch.norm(proj_rot_x[0, 0] - central_slice)
    
    print(f"Central slice theorem validation:")
    print(f"  Identity vs central slice: {diff_identity:.6f}")
    print(f"  Y-rotation vs central slice: {diff_rot_y:.6f}")
    print(f"  X-rotation vs central slice: {diff_rot_x:.6f}")
    
    # Identity should be closest to central slice
    assert diff_identity < diff_rot_y, "Identity projection should be closer to central slice than Y rotation"
    assert diff_identity < diff_rot_x, "Identity projection should be closer to central slice than X rotation"