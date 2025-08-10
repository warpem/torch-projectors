"""
Cross-platform compatibility tests for torch-projectors 2D->3D back-projection.

This module tests that CPU, CUDA, and MPS implementations produce identical results
for 2D->3D back-projection across all features including forward/backward passes and gradient computation.
"""

import torch
import torch.nn.functional as F
import torch_projectors
import pytest
import math
import sys
import os

# Add parent directory to path to import test_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_backproject_2d_to_3d_identical_comprehensive():
    """
    Comprehensive test that CPU and CUDA 2D->3D back-projection implementations produce identical results
    across all features: multiple projection sets, poses, shifts, interpolations,
    and full forward/backward passes with realistic loss computation.
    """
    torch.manual_seed(42)
    
    # Test parameters - comprehensive feature coverage
    num_projection_sets = 3
    num_poses = 4
    boxsize = 64
    proj_size = 48
    
    # Create comprehensive test data on CPU first
    dtype = torch.complex64
    
    # Multiple projection sets with different characteristics (2D projections for 3D back-projection)
    projections_cpu = torch.randn(num_projection_sets, num_poses, proj_size, proj_size//2 + 1, 
                                dtype=dtype, device='cpu').requires_grad_(True)
    
    # Optional weights for CTF handling
    weights_cpu = torch.rand(num_projection_sets, num_poses, proj_size, proj_size//2 + 1,
                           dtype=torch.float32, device='cpu').requires_grad_(True)
    
    # Multiple poses with varied 3D rotation angles and shifts
    angles = torch.tensor([0.0, 30.0, 60.0, 90.0], device='cpu') * math.pi / 180.0
    rotations_cpu = torch.zeros(num_projection_sets, num_poses, 3, 3, device='cpu')
    for b in range(num_projection_sets):
        for p in range(num_poses):
            # Vary angles across batches and poses (Z-axis rotations for simplicity)
            angle = angles[p % len(angles)] + b * 0.1  # Small batch-dependent variation
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            # 3x3 rotation matrix around Z-axis
            rotations_cpu[b, p] = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
    rotations_cpu.requires_grad_(True)
    
    # Comprehensive shifts - different per batch and pose
    shifts_cpu = torch.zeros(num_projection_sets, num_poses, 2, device='cpu')
    for b in range(num_projection_sets):
        for p in range(num_poses):
            # Create varied shift patterns
            shifts_cpu[b, p, 0] = (b - 1) * 2.0 + p * 0.5  # X shifts
            shifts_cpu[b, p, 1] = (p - 1.5) * 1.5           # Y shifts
    shifts_cpu.requires_grad_(True)
    
    # Copy all data to CUDA
    projections_cuda = projections_cpu.detach().clone().cuda().requires_grad_(True)
    weights_cuda = weights_cpu.detach().clone().cuda().requires_grad_(True)
    rotations_cuda = rotations_cpu.detach().clone().cuda().requires_grad_(True) 
    shifts_cuda = shifts_cpu.detach().clone().cuda().requires_grad_(True)
    
    # Test both interpolation methods
    for interpolation in ['linear', 'cubic']:
        print(f"\nTesting {interpolation} interpolation...")
        
        # Forward pass - CPU
        reconstruction_cpu, weight_reconstruction_cpu = torch_projectors.backproject_2d_to_3d_forw(
            projections_cpu, rotations=rotations_cpu, weights=weights_cpu, shifts=shifts_cpu,
            interpolation=interpolation
        )
        
        # Forward pass - CUDA  
        reconstruction_cuda, weight_reconstruction_cuda = torch_projectors.backproject_2d_to_3d_forw(
            projections_cuda, rotations=rotations_cuda, weights=weights_cuda, shifts=shifts_cuda,
            interpolation=interpolation
        )
        
        # Check forward pass results are identical
        reconstruction_cuda_cpu = reconstruction_cuda.cpu()
        weight_reconstruction_cuda_cpu = weight_reconstruction_cuda.cpu()
        
        forward_diff = torch.abs(reconstruction_cpu - reconstruction_cuda_cpu)
        max_forward_diff = torch.max(forward_diff).item()
        mean_forward_diff = torch.mean(forward_diff).item()
        
        weight_diff = torch.abs(weight_reconstruction_cpu - weight_reconstruction_cuda_cpu)
        max_weight_diff = torch.max(weight_diff).item()
        mean_weight_diff = torch.mean(weight_diff).item()
        
        print(f"Forward pass - Max diff: {max_forward_diff:.2e}, Mean diff: {mean_forward_diff:.2e}")
        print(f"Weight pass - Max diff: {max_weight_diff:.2e}, Mean diff: {mean_weight_diff:.2e}")
        
        # Use reasonable tolerance for single-precision floating point
        assert max_forward_diff < 1e-3, f"Forward pass differs too much: {max_forward_diff}"
        assert mean_forward_diff < 1e-4, f"Forward pass mean diff too high: {mean_forward_diff}"
        assert max_weight_diff < 1e-3, f"Weight pass differs too much: {max_weight_diff}"
        assert mean_weight_diff < 1e-4, f"Weight pass mean diff too high: {mean_weight_diff}"
        
        # Create realistic loss function for 3D back-projection
        def compute_realistic_3d_backproject_loss(reconstruction, weight_reconstruction):
            """Compute a realistic loss function for 3D back-projection results"""
            # 1. Data fidelity term (L2 loss against synthetic "target")
            target_amplitude = torch.abs(reconstruction.detach()) * 0.8  # Synthetic target
            amplitude_loss = F.mse_loss(torch.abs(reconstruction), target_amplitude)
            
            # 2. Total variation regularization (smoothness) on 3D reconstruction
            rec_real = reconstruction.real
            rec_imag = reconstruction.imag
            
            # Real part TV - all three spatial dimensions
            tv_r_d = torch.abs(rec_real[:, 1:, :, :] - rec_real[:, :-1, :, :])  # Depth
            tv_r_h = torch.abs(rec_real[:, :, 1:, :] - rec_real[:, :, :-1, :])  # Height
            tv_r_w = torch.abs(rec_real[:, :, :, 1:] - rec_real[:, :, :, :-1])  # Width
            tv_real = torch.mean(tv_r_d) + torch.mean(tv_r_h) + torch.mean(tv_r_w)
            
            # Imaginary part TV - all three spatial dimensions
            tv_i_d = torch.abs(rec_imag[:, 1:, :, :] - rec_imag[:, :-1, :, :])  # Depth
            tv_i_h = torch.abs(rec_imag[:, :, 1:, :] - rec_imag[:, :, :-1, :])  # Height
            tv_i_w = torch.abs(rec_imag[:, :, :, 1:] - rec_imag[:, :, :, :-1])  # Width
            tv_imag = torch.mean(tv_i_d) + torch.mean(tv_i_h) + torch.mean(tv_i_w)
            
            # 3. Weight consistency term (weights should be smooth and positive) - still 2D
            weight_penalty = torch.mean(torch.relu(-weight_reconstruction))  # Penalize negative weights
            weight_tv_h = torch.abs(weight_reconstruction[:, 1:, :] - weight_reconstruction[:, :-1, :])
            weight_tv_w = torch.abs(weight_reconstruction[:, :, 1:] - weight_reconstruction[:, :, :-1])
            weight_tv = torch.mean(weight_tv_h) + torch.mean(weight_tv_w)
            
            # Combine all loss terms
            total_loss = (amplitude_loss + 
                         0.001 * (tv_real + tv_imag) + 
                         0.001 * weight_penalty +
                         0.0001 * weight_tv)
            
            return total_loss
        
        # Compute loss and backward pass - CPU
        loss_cpu = compute_realistic_3d_backproject_loss(reconstruction_cpu, weight_reconstruction_cpu)
        loss_cpu.backward()
        
        # Get gradients from CPU
        proj_grad_cpu = projections_cpu.grad.clone()
        weight_grad_cpu = weights_cpu.grad.clone()
        rot_grad_cpu = rotations_cpu.grad.clone() 
        shift_grad_cpu = shifts_cpu.grad.clone()
        
        # Clear gradients
        projections_cpu.grad = None
        weights_cpu.grad = None
        rotations_cpu.grad = None
        shifts_cpu.grad = None
        
        # Compute loss and backward pass - CUDA
        loss_cuda = compute_realistic_3d_backproject_loss(reconstruction_cuda, weight_reconstruction_cuda)
        loss_cuda.backward()
        
        # Get gradients from CUDA and move to CPU for comparison
        proj_grad_cuda = projections_cuda.grad.cpu()
        weight_grad_cuda = weights_cuda.grad.cpu()
        rot_grad_cuda = rotations_cuda.grad.cpu()
        shift_grad_cuda = shifts_cuda.grad.cpu()
        
        # Clear gradients
        projections_cuda.grad = None
        weights_cuda.grad = None
        rotations_cuda.grad = None
        shifts_cuda.grad = None
        
        # Check loss values are identical
        loss_diff = torch.abs(loss_cpu - loss_cuda.cpu()).item()
        print(f"Loss difference: {loss_diff:.2e}")
        assert loss_diff < 1e-5, f"Loss differs too much: {loss_diff}"
        
        # Check projection gradients
        proj_grad_diff = torch.abs(proj_grad_cpu - proj_grad_cuda)
        max_proj_grad_diff = torch.max(proj_grad_diff).item()
        mean_proj_grad_diff = torch.mean(proj_grad_diff).item()
        print(f"Projection grad - Max diff: {max_proj_grad_diff:.2e}, Mean diff: {mean_proj_grad_diff:.2e}")
        assert max_proj_grad_diff < 1e-4, f"Projection gradients differ too much: {max_proj_grad_diff}"
        
        # Check weight gradients
        weight_grad_diff = torch.abs(weight_grad_cpu - weight_grad_cuda)
        max_weight_grad_diff = torch.max(weight_grad_diff).item()
        mean_weight_grad_diff = torch.mean(weight_grad_diff).item()
        print(f"Weight grad - Max diff: {max_weight_grad_diff:.2e}, Mean diff: {mean_weight_grad_diff:.2e}")
        assert max_weight_grad_diff < 1e-4, f"Weight gradients differ too much: {max_weight_grad_diff}"
        
        # Check rotation gradients
        rot_grad_diff = torch.abs(rot_grad_cpu - rot_grad_cuda)
        max_rot_grad_diff = torch.max(rot_grad_diff).item()
        mean_rot_grad_diff = torch.mean(rot_grad_diff).item()
        print(f"Rotation grad - Max diff: {max_rot_grad_diff:.2e}, Mean diff: {mean_rot_grad_diff:.2e}")
        assert max_rot_grad_diff < 1e-3, f"Rotation gradients differ too much: {max_rot_grad_diff}"
        
        # Check shift gradients  
        shift_grad_diff = torch.abs(shift_grad_cpu - shift_grad_cuda)
        max_shift_grad_diff = torch.max(shift_grad_diff).item()
        mean_shift_grad_diff = torch.mean(shift_grad_diff).item()
        print(f"Shift grad - Max diff: {max_shift_grad_diff:.2e}, Mean diff: {mean_shift_grad_diff:.2e}")
        assert max_shift_grad_diff < 1e-4, f"Shift gradients differ too much: {max_shift_grad_diff}"
        
        print(f"✓ {interpolation} interpolation: All CPU/CUDA 2D->3D back-projection results identical within tolerance")
    
    print(f"\n✓ Comprehensive CPU/CUDA 2D->3D back-projection comparison test passed!")
    print(f"  - {num_projection_sets} projection sets × {num_poses} poses tested")
    print(f"  - Both linear and cubic interpolation verified")
    print(f"  - 3x3 rotations, shifts, and weight accumulation tested")
    print(f"  - Full forward/backward pass with realistic 3D loss function")
    print(f"  - All gradients (projection, weights, rotation, shift) verified identical")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cpu_mps_backproject_2d_to_3d_identical_comprehensive():
    """
    Comprehensive test that CPU and MPS 2D->3D back-projection implementations produce identical results
    across all features: multiple projection sets, poses, shifts, interpolations,
    and full forward/backward passes with realistic loss computation.
    """
    torch.manual_seed(42)
    
    # Test parameters - comprehensive feature coverage
    num_projection_sets = 3
    num_poses = 4
    boxsize = 64
    proj_size = 48
    
    # Create comprehensive test data on CPU first
    dtype = torch.complex64
    
    # Multiple projection sets with different characteristics (2D projections for 3D back-projection)
    projections_cpu = torch.randn(num_projection_sets, num_poses, proj_size, proj_size//2 + 1, 
                                dtype=dtype, device='cpu').requires_grad_(True)
    
    # Optional weights for CTF handling
    weights_cpu = torch.rand(num_projection_sets, num_poses, proj_size, proj_size//2 + 1,
                           dtype=torch.float32, device='cpu').requires_grad_(True)
    
    # Multiple poses with varied 3D rotation angles and shifts
    angles = torch.tensor([0.0, 30.0, 60.0, 90.0], device='cpu') * math.pi / 180.0
    rotations_cpu = torch.zeros(num_projection_sets, num_poses, 3, 3, device='cpu')
    for b in range(num_projection_sets):
        for p in range(num_poses):
            # Vary angles across batches and poses (Z-axis rotations for simplicity)
            angle = angles[p % len(angles)] + b * 0.1  # Small batch-dependent variation
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            # 3x3 rotation matrix around Z-axis
            rotations_cpu[b, p] = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
    rotations_cpu.requires_grad_(True)
    
    # Comprehensive shifts - different per batch and pose
    shifts_cpu = torch.zeros(num_projection_sets, num_poses, 2, device='cpu')
    for b in range(num_projection_sets):
        for p in range(num_poses):
            # Create varied shift patterns
            shifts_cpu[b, p, 0] = (b - 1) * 2.0 + p * 0.5  # X shifts
            shifts_cpu[b, p, 1] = (p - 1.5) * 1.5           # Y shifts
    shifts_cpu.requires_grad_(True)
    
    # Copy all data to MPS
    projections_mps = projections_cpu.detach().clone().to('mps').requires_grad_(True)
    weights_mps = weights_cpu.detach().clone().to('mps').requires_grad_(True)
    rotations_mps = rotations_cpu.detach().clone().to('mps').requires_grad_(True) 
    shifts_mps = shifts_cpu.detach().clone().to('mps').requires_grad_(True)
    
    # Test both interpolation methods
    for interpolation in ['linear', 'cubic']:
        print(f"\nTesting {interpolation} interpolation...")
        
        # Forward pass - CPU
        reconstruction_cpu, weight_reconstruction_cpu = torch_projectors.backproject_2d_to_3d_forw(
            projections_cpu, rotations=rotations_cpu, weights=weights_cpu, shifts=shifts_cpu,
            interpolation=interpolation
        )
        
        # Forward pass - MPS  
        reconstruction_mps, weight_reconstruction_mps = torch_projectors.backproject_2d_to_3d_forw(
            projections_mps, rotations=rotations_mps, weights=weights_mps, shifts=shifts_mps,
            interpolation=interpolation
        )
        
        # Check forward pass results are identical
        reconstruction_mps_cpu = reconstruction_mps.cpu()
        weight_reconstruction_mps_cpu = weight_reconstruction_mps.cpu()
        
        forward_diff = torch.abs(reconstruction_cpu - reconstruction_mps_cpu)
        max_forward_diff = torch.max(forward_diff).item()
        mean_forward_diff = torch.mean(forward_diff).item()
        
        weight_diff = torch.abs(weight_reconstruction_cpu - weight_reconstruction_mps_cpu)
        max_weight_diff = torch.max(weight_diff).item()
        mean_weight_diff = torch.mean(weight_diff).item()
        
        print(f"Forward pass - Max diff: {max_forward_diff:.2e}, Mean diff: {mean_forward_diff:.2e}")
        print(f"Weight pass - Max diff: {max_weight_diff:.2e}, Mean diff: {mean_weight_diff:.2e}")
        
        # Use reasonable tolerance for single-precision floating point
        assert max_forward_diff < 1e-3, f"Forward pass differs too much: {max_forward_diff}"
        assert mean_forward_diff < 1e-4, f"Forward pass mean diff too high: {mean_forward_diff}"
        assert max_weight_diff < 1e-3, f"Weight pass differs too much: {max_weight_diff}"
        assert mean_weight_diff < 1e-4, f"Weight pass mean diff too high: {mean_weight_diff}"
        
        # Create realistic loss function for 3D back-projection
        def compute_realistic_3d_backproject_loss(reconstruction, weight_reconstruction):
            """Compute a realistic loss function for 3D back-projection results"""
            # 1. Data fidelity term (L2 loss against synthetic "target")
            target_amplitude = torch.abs(reconstruction.detach()) * 0.8  # Synthetic target
            amplitude_loss = F.mse_loss(torch.abs(reconstruction), target_amplitude)
            
            # 2. Total variation regularization (smoothness) on 3D reconstruction
            rec_real = reconstruction.real
            rec_imag = reconstruction.imag
            
            # Real part TV - all three spatial dimensions
            tv_r_d = torch.abs(rec_real[:, 1:, :, :] - rec_real[:, :-1, :, :])  # Depth
            tv_r_h = torch.abs(rec_real[:, :, 1:, :] - rec_real[:, :, :-1, :])  # Height
            tv_r_w = torch.abs(rec_real[:, :, :, 1:] - rec_real[:, :, :, :-1])  # Width
            tv_real = torch.mean(tv_r_d) + torch.mean(tv_r_h) + torch.mean(tv_r_w)
            
            # Imaginary part TV - all three spatial dimensions
            tv_i_d = torch.abs(rec_imag[:, 1:, :, :] - rec_imag[:, :-1, :, :])  # Depth
            tv_i_h = torch.abs(rec_imag[:, :, 1:, :] - rec_imag[:, :, :-1, :])  # Height
            tv_i_w = torch.abs(rec_imag[:, :, :, 1:] - rec_imag[:, :, :, :-1])  # Width
            tv_imag = torch.mean(tv_i_d) + torch.mean(tv_i_h) + torch.mean(tv_i_w)
            
            # 3. Weight consistency term (weights should be smooth and positive) - still 2D
            weight_penalty = torch.mean(torch.relu(-weight_reconstruction))  # Penalize negative weights
            weight_tv_h = torch.abs(weight_reconstruction[:, 1:, :] - weight_reconstruction[:, :-1, :])
            weight_tv_w = torch.abs(weight_reconstruction[:, :, 1:] - weight_reconstruction[:, :, :-1])
            weight_tv = torch.mean(weight_tv_h) + torch.mean(weight_tv_w)
            
            # Combine all loss terms
            total_loss = (amplitude_loss + 
                         0.001 * (tv_real + tv_imag) + 
                         0.001 * weight_penalty +
                         0.0001 * weight_tv)
            
            return total_loss
        
        # Compute loss and backward pass - CPU
        loss_cpu = compute_realistic_3d_backproject_loss(reconstruction_cpu, weight_reconstruction_cpu)
        loss_cpu.backward()
        
        # Get gradients from CPU
        proj_grad_cpu = projections_cpu.grad.clone()
        weight_grad_cpu = weights_cpu.grad.clone()
        rot_grad_cpu = rotations_cpu.grad.clone() 
        shift_grad_cpu = shifts_cpu.grad.clone()
        
        # Clear gradients
        projections_cpu.grad = None
        weights_cpu.grad = None
        rotations_cpu.grad = None
        shifts_cpu.grad = None
        
        # Compute loss and backward pass - MPS
        loss_mps = compute_realistic_3d_backproject_loss(reconstruction_mps, weight_reconstruction_mps)
        loss_mps.backward()
        
        # Get gradients from MPS and move to CPU for comparison
        proj_grad_mps = projections_mps.grad.cpu()
        weight_grad_mps = weights_mps.grad.cpu()
        rot_grad_mps = rotations_mps.grad.cpu()
        shift_grad_mps = shifts_mps.grad.cpu()
        
        # Clear gradients
        projections_mps.grad = None
        weights_mps.grad = None
        rotations_mps.grad = None
        shifts_mps.grad = None
        
        # Check loss values are identical
        loss_diff = torch.abs(loss_cpu - loss_mps.cpu()).item()
        print(f"Loss difference: {loss_diff:.2e}")
        assert loss_diff < 1e-5, f"Loss differs too much: {loss_diff}"
        
        # Check projection gradients
        proj_grad_diff = torch.abs(proj_grad_cpu - proj_grad_mps)
        max_proj_grad_diff = torch.max(proj_grad_diff).item()
        mean_proj_grad_diff = torch.mean(proj_grad_diff).item()
        print(f"Projection grad - Max diff: {max_proj_grad_diff:.2e}, Mean diff: {mean_proj_grad_diff:.2e}")
        assert max_proj_grad_diff < 1e-4, f"Projection gradients differ too much: {max_proj_grad_diff}"
        
        # Check weight gradients
        weight_grad_diff = torch.abs(weight_grad_cpu - weight_grad_mps)
        max_weight_grad_diff = torch.max(weight_grad_diff).item()
        mean_weight_grad_diff = torch.mean(weight_grad_diff).item()
        print(f"Weight grad - Max diff: {max_weight_grad_diff:.2e}, Mean diff: {mean_weight_grad_diff:.2e}")
        assert max_weight_grad_diff < 1e-4, f"Weight gradients differ too much: {max_weight_grad_diff}"
        
        # Check rotation gradients
        rot_grad_diff = torch.abs(rot_grad_cpu - rot_grad_mps)
        max_rot_grad_diff = torch.max(rot_grad_diff).item()
        mean_rot_grad_diff = torch.mean(rot_grad_diff).item()
        print(f"Rotation grad - Max diff: {max_rot_grad_diff:.2e}, Mean diff: {mean_rot_grad_diff:.2e}")
        assert max_rot_grad_diff < 1e-3, f"Rotation gradients differ too much: {max_rot_grad_diff}"
        
        # Check shift gradients  
        shift_grad_diff = torch.abs(shift_grad_cpu - shift_grad_mps)
        max_shift_grad_diff = torch.max(shift_grad_diff).item()
        mean_shift_grad_diff = torch.mean(shift_grad_diff).item()
        print(f"Shift grad - Max diff: {max_shift_grad_diff:.2e}, Mean diff: {mean_shift_grad_diff:.2e}")
        assert max_shift_grad_diff < 1e-4, f"Shift gradients differ too much: {max_shift_grad_diff}"
        
        print(f"✓ {interpolation} interpolation: All CPU/MPS 2D->3D back-projection results identical within tolerance")
    
    print(f"\n✓ Comprehensive CPU/MPS 2D->3D back-projection comparison test passed!")
    print(f"  - {num_projection_sets} projection sets × {num_poses} poses tested")
    print(f"  - Both linear and cubic interpolation verified")
    print(f"  - 3x3 rotations, shifts, and weight accumulation tested")
    print(f"  - Full forward/backward pass with realistic 3D loss function")
    print(f"  - All gradients (projection, weights, rotation, shift) verified identical")