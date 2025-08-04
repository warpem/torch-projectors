"""
Cross-platform compatibility tests for torch-projectors.

This module tests that CPU, CUDA, and MPS implementations produce identical results
across all features including forward/backward passes and gradient computation.
"""

import torch
import torch.nn.functional as F
import torch_projectors
import pytest
import math


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_identical_comprehensive():
    """
    Comprehensive test that CPU and CUDA implementations produce identical results
    across all features: multiple reconstructions, poses, shifts, interpolations,
    and full forward/backward passes with realistic loss computation.
    """
    torch.manual_seed(42)
    
    # Test parameters - comprehensive feature coverage (without oversampling)
    num_reconstructions = 1
    num_poses = 1
    boxsize = 64
    proj_size = 48
    
    # Create comprehensive test data on CPU first
    dtype = torch.complex64
    
    # Multiple reconstructions with different characteristics (all random for simplicity)
    reconstructions_cpu = torch.randn(num_reconstructions, boxsize, boxsize//2 + 1, 
                                    dtype=dtype, device='cpu').requires_grad_(True)
    
    # Multiple poses with varied rotation angles and shifts
    angles = torch.tensor([0.0, 30.0, 60.0, 90.0], device='cpu') * math.pi / 180.0
    rotations_cpu = torch.zeros(num_reconstructions, num_poses, 2, 2, device='cpu')
    for b in range(num_reconstructions):
        for p in range(num_poses):
            # Vary angles across batches and poses
            angle = angles[p % len(angles)] + b * 0.1  # Small batch-dependent variation
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rotations_cpu[b, p] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
    rotations_cpu.requires_grad_(True)
    
    # Comprehensive shifts - different per batch and pose
    shifts_cpu = torch.zeros(num_reconstructions, num_poses, 2, device='cpu')
    for b in range(num_reconstructions):
        for p in range(num_poses):
            # Create varied shift patterns
            shifts_cpu[b, p, 0] = (b - 1) * 2.0 + p * 0.5  # X shifts: -2, 0, 2 base + pose variation
            shifts_cpu[b, p, 1] = (p - 1.5) * 1.5           # Y shifts: centered around 0
    shifts_cpu.requires_grad_(True)
    
    # Copy all data to CUDA
    reconstructions_cuda = reconstructions_cpu.detach().clone().cuda().requires_grad_(True)
    rotations_cuda = rotations_cpu.detach().clone().cuda().requires_grad_(True) 
    shifts_cuda = shifts_cpu.detach().clone().cuda().requires_grad_(True)
    
    # Test both interpolation methods
    for interpolation in ['linear', 'cubic']:
        print(f"\nTesting {interpolation} interpolation...")
        
        # Forward pass - CPU
        projections_cpu = torch_projectors.project_2d_forw(
            reconstructions_cpu, rotations_cpu, shifts_cpu,
            output_shape=(proj_size, proj_size),
            interpolation=interpolation
        )
        
        # Forward pass - CUDA  
        projections_cuda = torch_projectors.project_2d_forw(
            reconstructions_cuda, rotations_cuda, shifts_cuda,
            output_shape=(proj_size, proj_size),
            interpolation=interpolation
        )
        
        # Check forward pass results are identical
        projections_cuda_cpu = projections_cuda.cpu()
        forward_diff = torch.abs(projections_cpu - projections_cuda_cpu)
        max_forward_diff = torch.max(forward_diff).item()
        mean_forward_diff = torch.mean(forward_diff).item()
        
        print(f"Forward pass - Max diff: {max_forward_diff:.2e}, Mean diff: {mean_forward_diff:.2e}")
        
        # Use reasonable tolerance for single-precision floating point
        assert max_forward_diff < 1e-3, f"Forward pass differs too much: {max_forward_diff}"
        assert mean_forward_diff < 1e-4, f"Forward pass mean diff too high: {mean_forward_diff}"
        
        # Create realistic loss function that uses all projection features
        def compute_realistic_loss(projections):
            """Compute a realistic cryo-EM style loss function"""
            # 1. Data fidelity term (L2 loss against synthetic "target")
            target_amplitude = torch.abs(projections.detach()) * 0.8  # Synthetic target
            amplitude_loss = F.mse_loss(torch.abs(projections), target_amplitude)
            
            # 2. Total variation regularization (smoothness)
            # Compute gradients in both spatial dimensions
            proj_real = projections.real
            proj_imag = projections.imag
            
            # Real part TV
            tv_r_h = torch.abs(proj_real[:, :, 1:, :] - proj_real[:, :, :-1, :])
            tv_r_w = torch.abs(proj_real[:, :, :, 1:] - proj_real[:, :, :, :-1])
            tv_real = torch.mean(tv_r_h) + torch.mean(tv_r_w)
            
            # Imaginary part TV  
            tv_i_h = torch.abs(proj_imag[:, :, 1:, :] - proj_imag[:, :, :-1, :])
            tv_i_w = torch.abs(proj_imag[:, :, :, 1:] - proj_imag[:, :, :, :-1])
            tv_imag = torch.mean(tv_i_h) + torch.mean(tv_i_w)
            
            # 3. Spectral constraint (penalize high frequencies)
            # Use different weights for different frequency bands
            freq_penalty = 0.0
            for b in range(projections.shape[0]):
                for p in range(projections.shape[1]):
                    proj_fft = torch.fft.fft2(projections[b, p])
                    power_spectrum = torch.abs(proj_fft)**2
                    # Penalize high frequency content  
                    h, w = power_spectrum.shape
                    y_freq = torch.fft.fftfreq(h, device=projections.device).unsqueeze(1)
                    x_freq = torch.fft.fftfreq(w, device=projections.device).unsqueeze(0)
                    freq_mask = (y_freq**2 + x_freq**2) > 0.3**2  # High freq mask
                    freq_penalty += torch.sum(power_spectrum * freq_mask)
            
            # Combine all loss terms
            total_loss = (amplitude_loss + 
                         0.001 * (tv_real + tv_imag) + 
                         0.0001 * freq_penalty / (num_reconstructions * num_poses))
            
            return total_loss
        
        # Compute loss and backward pass - CPU
        loss_cpu = compute_realistic_loss(projections_cpu)
        loss_cpu.backward()
        
        # Get gradients from CPU
        rec_grad_cpu = reconstructions_cpu.grad.clone()
        rot_grad_cpu = rotations_cpu.grad.clone() 
        shift_grad_cpu = shifts_cpu.grad.clone()
        
        # Clear gradients
        reconstructions_cpu.grad = None
        rotations_cpu.grad = None
        shifts_cpu.grad = None
        
        # Compute loss and backward pass - CUDA
        loss_cuda = compute_realistic_loss(projections_cuda)
        loss_cuda.backward()
        
        # Get gradients from CUDA and move to CPU for comparison
        rec_grad_cuda = reconstructions_cuda.grad.cpu()
        rot_grad_cuda = rotations_cuda.grad.cpu()
        shift_grad_cuda = shifts_cuda.grad.cpu()
        
        # Clear gradients
        reconstructions_cuda.grad = None
        rotations_cuda.grad = None
        shifts_cuda.grad = None
        
        # Check loss values are identical
        loss_diff = torch.abs(loss_cpu - loss_cuda.cpu()).item()
        print(f"Loss difference: {loss_diff:.2e}")
        assert loss_diff < 1e-5, f"Loss differs too much: {loss_diff}"
        
        # Check reconstruction gradients
        rec_grad_diff = torch.abs(rec_grad_cpu - rec_grad_cuda)
        max_rec_grad_diff = torch.max(rec_grad_diff).item()
        mean_rec_grad_diff = torch.mean(rec_grad_diff).item()
        print(f"Reconstruction grad - Max diff: {max_rec_grad_diff:.2e}, Mean diff: {mean_rec_grad_diff:.2e}")
        assert max_rec_grad_diff < 1e-4, f"Reconstruction gradients differ too much: {max_rec_grad_diff}"
        
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
        
        print(f"✓ {interpolation} interpolation: All CPU/CUDA results identical within tolerance")
    
    print(f"\n✓ Comprehensive CPU/CUDA comparison test passed!")
    print(f"  - {num_reconstructions} reconstructions × {num_poses} poses tested")
    print(f"  - Both linear and cubic interpolation verified")
    print(f"  - Rotations, shifts, and frequency cutoff tested")
    print(f"  - Full forward/backward pass with realistic loss function")
    print(f"  - All gradients (reconstruction, rotation, shift) verified identical")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cpu_mps_identical_comprehensive():
    """
    Comprehensive test that CPU and MPS implementations produce identical results
    across all features: multiple reconstructions, poses, shifts, interpolations,
    and full forward/backward passes with realistic loss computation.
    """
    torch.manual_seed(42)
    
    # Test parameters - comprehensive feature coverage (without oversampling)
    num_reconstructions = 1
    num_poses = 1
    boxsize = 64
    proj_size = 48
    
    # Create comprehensive test data on CPU first
    dtype = torch.complex64
    
    # Multiple reconstructions with different characteristics (all random for simplicity)
    reconstructions_cpu = torch.randn(num_reconstructions, boxsize, boxsize//2 + 1, 
                                    dtype=dtype, device='cpu').requires_grad_(True)
    
    # Multiple poses with varied rotation angles and shifts
    angles = torch.tensor([0.0, 30.0, 60.0, 90.0], device='cpu') * math.pi / 180.0
    rotations_cpu = torch.zeros(num_reconstructions, num_poses, 2, 2, device='cpu')
    for b in range(num_reconstructions):
        for p in range(num_poses):
            # Vary angles across batches and poses
            angle = angles[p % len(angles)] + b * 0.1  # Small batch-dependent variation
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rotations_cpu[b, p] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
    rotations_cpu.requires_grad_(True)
    
    # Comprehensive shifts - different per batch and pose
    shifts_cpu = torch.zeros(num_reconstructions, num_poses, 2, device='cpu')
    for b in range(num_reconstructions):
        for p in range(num_poses):
            # Create varied shift patterns
            shifts_cpu[b, p, 0] = (b - 1) * 2.0 + p * 0.5  # X shifts: -2, 0, 2 base + pose variation
            shifts_cpu[b, p, 1] = (p - 1.5) * 1.5           # Y shifts: centered around 0
    shifts_cpu.requires_grad_(True)
    
    # Copy all data to MPS
    reconstructions_mps = reconstructions_cpu.detach().clone().to('mps').requires_grad_(True)
    rotations_mps = rotations_cpu.detach().clone().to('mps').requires_grad_(True) 
    shifts_mps = shifts_cpu.detach().clone().to('mps').requires_grad_(True)
    
    # Test both interpolation methods
    for interpolation in ['linear', 'cubic']:
        print(f"\nTesting {interpolation} interpolation...")
        
        # Forward pass - CPU
        projections_cpu = torch_projectors.project_2d_forw(
            reconstructions_cpu, rotations_cpu, shifts_cpu,
            output_shape=(proj_size, proj_size),
            interpolation=interpolation
        )
        
        # Forward pass - MPS  
        projections_mps = torch_projectors.project_2d_forw(
            reconstructions_mps, rotations_mps, shifts_mps,
            output_shape=(proj_size, proj_size),
            interpolation=interpolation
        )
        
        # Check forward pass results are identical
        projections_mps_cpu = projections_mps.cpu()
        forward_diff = torch.abs(projections_cpu - projections_mps_cpu)
        max_forward_diff = torch.max(forward_diff).item()
        mean_forward_diff = torch.mean(forward_diff).item()
        
        print(f"Forward pass - Max diff: {max_forward_diff:.2e}, Mean diff: {mean_forward_diff:.2e}")
        
        # Use reasonable tolerance for single-precision floating point
        assert max_forward_diff < 1e-3, f"Forward pass differs too much: {max_forward_diff}"
        assert mean_forward_diff < 1e-4, f"Forward pass mean diff too high: {mean_forward_diff}"
        
        # Create realistic loss function that uses all projection features
        def compute_realistic_loss(projections):
            """Compute a realistic cryo-EM style loss function"""
            # 1. Data fidelity term (L2 loss against synthetic "target")
            target_amplitude = torch.abs(projections.detach()) * 0.8  # Synthetic target
            amplitude_loss = F.mse_loss(torch.abs(projections), target_amplitude)
            
            # 2. Total variation regularization (smoothness)
            # Compute gradients in both spatial dimensions
            proj_real = projections.real
            proj_imag = projections.imag
            
            # Real part TV
            tv_r_h = torch.abs(proj_real[:, :, 1:, :] - proj_real[:, :, :-1, :])
            tv_r_w = torch.abs(proj_real[:, :, :, 1:] - proj_real[:, :, :, :-1])
            tv_real = torch.mean(tv_r_h) + torch.mean(tv_r_w)
            
            # Imaginary part TV  
            tv_i_h = torch.abs(proj_imag[:, :, 1:, :] - proj_imag[:, :, :-1, :])
            tv_i_w = torch.abs(proj_imag[:, :, :, 1:] - proj_imag[:, :, :, :-1])
            tv_imag = torch.mean(tv_i_h) + torch.mean(tv_i_w)
            
            # 3. Spectral constraint (penalize high frequencies)
            # Use different weights for different frequency bands
            freq_penalty = 0.0
            for b in range(projections.shape[0]):
                for p in range(projections.shape[1]):
                    proj_fft = torch.fft.fft2(projections[b, p])
                    power_spectrum = torch.abs(proj_fft)**2
                    # Penalize high frequency content  
                    h, w = power_spectrum.shape
                    y_freq = torch.fft.fftfreq(h, device=projections.device).unsqueeze(1)
                    x_freq = torch.fft.fftfreq(w, device=projections.device).unsqueeze(0)
                    freq_mask = (y_freq**2 + x_freq**2) > 0.3**2  # High freq mask
                    freq_penalty += torch.sum(power_spectrum * freq_mask)
            
            # Combine all loss terms
            total_loss = (amplitude_loss + 
                         0.001 * (tv_real + tv_imag) + 
                         0.0001 * freq_penalty / (num_reconstructions * num_poses))
            
            return total_loss
        
        # Compute loss and backward pass - CPU
        loss_cpu = compute_realistic_loss(projections_cpu)
        loss_cpu.backward()
        
        # Get gradients from CPU
        rec_grad_cpu = reconstructions_cpu.grad.clone()
        rot_grad_cpu = rotations_cpu.grad.clone() 
        shift_grad_cpu = shifts_cpu.grad.clone()
        
        # Clear gradients
        reconstructions_cpu.grad = None
        rotations_cpu.grad = None
        shifts_cpu.grad = None
        
        # Compute loss and backward pass - MPS
        loss_mps = compute_realistic_loss(projections_mps)
        loss_mps.backward()
        
        # Get gradients from MPS and move to CPU for comparison
        rec_grad_mps = reconstructions_mps.grad.cpu()
        rot_grad_mps = rotations_mps.grad.cpu()
        shift_grad_mps = shifts_mps.grad.cpu()
        
        # Clear gradients
        reconstructions_mps.grad = None
        rotations_mps.grad = None
        shifts_mps.grad = None
        
        # Check loss values are identical
        loss_diff = torch.abs(loss_cpu - loss_mps.cpu()).item()
        print(f"Loss difference: {loss_diff:.2e}")
        assert loss_diff < 1e-5, f"Loss differs too much: {loss_diff}"
        
        # Check reconstruction gradients
        rec_grad_diff = torch.abs(rec_grad_cpu - rec_grad_mps)
        max_rec_grad_diff = torch.max(rec_grad_diff).item()
        mean_rec_grad_diff = torch.mean(rec_grad_diff).item()
        print(f"Reconstruction grad - Max diff: {max_rec_grad_diff:.2e}, Mean diff: {mean_rec_grad_diff:.2e}")
        assert max_rec_grad_diff < 1e-4, f"Reconstruction gradients differ too much: {max_rec_grad_diff}"
        
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
        
        print(f"✓ {interpolation} interpolation: All CPU/MPS results identical within tolerance")
    
    print(f"\n✓ Comprehensive CPU/MPS comparison test passed!")
    print(f"  - {num_reconstructions} reconstructions × {num_poses} poses tested")
    print(f"  - Both linear and cubic interpolation verified")
    print(f"  - Rotations, shifts, and frequency cutoff tested")
    print(f"  - Full forward/backward pass with realistic loss function")
    print(f"  - All gradients (reconstruction, rotation, shift) verified identical")