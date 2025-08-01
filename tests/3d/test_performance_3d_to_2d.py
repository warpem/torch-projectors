"""
Performance and benchmarking tests for 3D->2D projections.

This module tests performance characteristics and compares interpolation quality
for the forward_project_3d_to_2d operation.
"""

import torch
import torch_projectors
import pytest
import math
import time
import statistics
from test_utils import device, plot_fourier_tensors


def test_performance_benchmark_3d_to_2d(device):
    """
    Performance benchmark comparing linear vs cubic interpolation for 3D->2D projections.
    
    Tests forward and backward projection performance with:
    - Multiple 3D reconstructions (batch processing)
    - Random 3D rotation matrices
    - Proper benchmarking practices (warmup, multiple runs, statistics)
    """
    
    # Benchmark parameters
    num_reconstructions = 8  # Fewer than 2D due to 3D complexity
    num_projections_per_rec = 512  # Fewer than 2D
    D, H, W = 128, 128, 128  # Cubical volumes
    W_half = W // 2 + 1
    num_warmup_runs = 3
    num_timing_runs = 10
    
    print(f"\n3D->2D Performance Benchmark:")
    print(f"- Reconstructions: {num_reconstructions}")
    print(f"- Projections per reconstruction: {num_projections_per_rec}")
    print(f"- Total projections: {num_reconstructions * num_projections_per_rec}")
    print(f"- Volume size: {D}x{H}x{W} -> {H}x{W}")
    print(f"- Warmup runs: {num_warmup_runs}, Timing runs: {num_timing_runs}")
    
    # Generate test data
    torch.manual_seed(42)
    reconstructions = torch.randn(num_reconstructions, D, H, W_half, dtype=torch.complex64, device=device)
    
    # Random 3D rotations - using Euler angles for variety
    angles_x = torch.rand(num_reconstructions, num_projections_per_rec, device=device) * 2 * math.pi
    angles_y = torch.rand(num_reconstructions, num_projections_per_rec, device=device) * 2 * math.pi
    angles_z = torch.rand(num_reconstructions, num_projections_per_rec, device=device) * 2 * math.pi
    
    rotations = torch.zeros(num_reconstructions, num_projections_per_rec, 3, 3, device=device)
    for i in range(num_reconstructions):
        for j in range(num_projections_per_rec):
            # Create rotation matrices around each axis
            cos_x, sin_x = torch.cos(angles_x[i, j]), torch.sin(angles_x[i, j])
            cos_y, sin_y = torch.cos(angles_y[i, j]), torch.sin(angles_y[i, j])
            cos_z, sin_z = torch.cos(angles_z[i, j]), torch.sin(angles_z[i, j])
            
            # Rotation around X axis
            Rx = torch.tensor([
                [1, 0, 0],
                [0, cos_x, -sin_x],
                [0, sin_x, cos_x]
            ], dtype=torch.float32, device=device)
            
            # Rotation around Y axis  
            Ry = torch.tensor([
                [cos_y, 0, sin_y],
                [0, 1, 0],
                [-sin_y, 0, cos_y]
            ], dtype=torch.float32, device=device)
            
            # Rotation around Z axis
            Rz = torch.tensor([
                [cos_z, -sin_z, 0],
                [sin_z, cos_z, 0],
                [0, 0, 1]
            ], dtype=torch.float32, device=device)
            
            # Combined rotation: Rz * Ry * Rx
            rotations[i, j] = Rz @ Ry @ Rx
    
    shifts = torch.randn(num_reconstructions, num_projections_per_rec, 2, device=device) * 5.0
    
    def benchmark_interpolation(interpolation_method):
        """Benchmark a specific interpolation method"""
        forward_times = []
        backward_times = []
        
        # Warmup runs
        for _ in range(num_warmup_runs):
            reconstructions.requires_grad_(True)
            projections = torch_projectors.forward_project_3d_to_2d(
                reconstructions, rotations, shifts, 
                output_shape=(H, W), interpolation=interpolation_method
            )
            loss = torch.sum(torch.abs(projections)**2)
            loss.backward()
            reconstructions.grad = None
        
        # Timing runs
        for _ in range(num_timing_runs):
            reconstructions.requires_grad_(True)
            
            # Time forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            projections = torch_projectors.forward_project_3d_to_2d(
                reconstructions, rotations, shifts,
                output_shape=(H, W), interpolation=interpolation_method
            )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            forward_time = time.perf_counter() - start_time
            forward_times.append(forward_time)
            
            # Time backward pass
            loss = torch.sum(torch.abs(projections)**2)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            start_time = time.perf_counter()
            
            loss.backward()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            backward_time = time.perf_counter() - start_time
            backward_times.append(backward_time)
            
            reconstructions.grad = None
        
        return forward_times, backward_times
    
    # Benchmark both methods
    print(f"\nBenchmarking linear interpolation...")
    linear_forward, linear_backward = benchmark_interpolation('linear')
    
    print(f"Benchmarking cubic interpolation...")
    cubic_forward, cubic_backward = benchmark_interpolation('cubic')
    
    # Calculate statistics
    def calc_stats(times):
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times)
        }
    
    linear_forward_stats = calc_stats(linear_forward)
    linear_backward_stats = calc_stats(linear_backward)
    cubic_forward_stats = calc_stats(cubic_forward)
    cubic_backward_stats = calc_stats(cubic_backward)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"3D->2D PERFORMANCE RESULTS")
    print(f"{'='*60}")
    
    print(f"\nForward Projection Times (seconds):")
    print(f"Linear  - Mean: {linear_forward_stats['mean']:.4f} ± {linear_forward_stats['stdev']:.4f}")
    print(f"Cubic   - Mean: {cubic_forward_stats['mean']:.4f} ± {cubic_forward_stats['stdev']:.4f}")
    print(f"Slowdown: {cubic_forward_stats['mean'] / linear_forward_stats['mean']:.2f}x")
    
    print(f"\nBackward Projection Times (seconds):")
    print(f"Linear  - Mean: {linear_backward_stats['mean']:.4f} ± {linear_backward_stats['stdev']:.4f}")
    print(f"Cubic   - Mean: {cubic_backward_stats['mean']:.4f} ± {cubic_backward_stats['stdev']:.4f}")
    print(f"Slowdown: {cubic_backward_stats['mean'] / linear_backward_stats['mean']:.2f}x")
    
    print(f"\nTotal Times (Forward + Backward):")
    linear_total = linear_forward_stats['mean'] + linear_backward_stats['mean']
    cubic_total = cubic_forward_stats['mean'] + cubic_backward_stats['mean']
    print(f"Linear  - {linear_total:.4f} seconds")
    print(f"Cubic   - {cubic_total:.4f} seconds")
    print(f"Slowdown: {cubic_total / linear_total:.2f}x")
    
    print(f"\nThroughput (projections/second):")
    total_projections = num_reconstructions * num_projections_per_rec
    print(f"Linear  - {total_projections / linear_total:.1f}")
    print(f"Cubic   - {total_projections / cubic_total:.1f}")


def test_interpolation_quality_comparison_3d_to_2d(device):
    """
    Compare linear vs cubic interpolation quality using round-trip 3D->2D->2D projection.
    
    Test procedure:
    1. Start with random 3D reconstruction  
    2. Project to 2D at identity rotation as reference
    3. Project 3D->2D at rotation, then 2D->2D back with inverse rotation (round-trip)
    4. Test multiple rotation axes (X, Y, Z) to verify interpolation works well in all dimensions
    5. Compare both interpolation methods, verify cubic performs better
    """
    torch.manual_seed(44)
    
    # Create random 3D reconstruction in Fourier space
    D, H, W = 32, 32, 32  # Cubical volumes
    W_half = W // 2 + 1
    reconstruction_3d = torch.randn(D, H, W_half, dtype=torch.complex64, device='cpu').to(device)
    
    # Identity projection as reference (3D->2D at identity rotation)
    identity_rotation_3d = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    reference_proj = torch_projectors.forward_project_3d_to_2d(
        reconstruction_3d.unsqueeze(0), 
        identity_rotation_3d, 
        output_shape=(H, W),
        interpolation='linear'  # Doesn't matter for identity
    )
    
    # Test rotations around different axes
    test_angles = [
        ("X-axis", [1, 0, 0]),
        ("Y-axis", [0, 1, 0]), 
        ("Z-axis", [0, 0, 1])
    ]
    
    angle_rad = math.radians(2.0)  # Small angle to test interpolation quality
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    linear_errors = []
    cubic_errors = []
    
    for axis_name, axis in test_angles:
        print(f"\nTesting round-trip interpolation around {axis_name}:")
        
        # Create rotation matrices around the specified axis
        if axis == [1, 0, 0]:  # X-axis
            rot_forward = torch.tensor([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            rot_back_2d = torch.tensor([
                [cos_a, sin_a], 
                [-sin_a, cos_a]
            ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        elif axis == [0, 1, 0]:  # Y-axis  
            rot_forward = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            rot_back_2d = torch.tensor([
                [cos_a, sin_a],
                [-sin_a, cos_a]
            ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        else:  # Z-axis
            rot_forward = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            rot_back_2d = torch.tensor([
                [cos_a, sin_a],
                [-sin_a, cos_a]
            ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        # Round-trip with linear interpolation: 3D->2D then 2D->2D back
        proj_rotated_linear = torch_projectors.forward_project_3d_to_2d(
            reconstruction_3d.unsqueeze(0), rot_forward, 
            output_shape=(H, W), interpolation='linear'
        )
        roundtrip_linear = torch_projectors.forward_project_2d(
            proj_rotated_linear.squeeze(1), rot_back_2d, 
            output_shape=(H, H), interpolation='linear'
        )
        
        # Round-trip with cubic interpolation: 3D->2D then 2D->2D back
        proj_rotated_cubic = torch_projectors.forward_project_3d_to_2d(
            reconstruction_3d.unsqueeze(0), rot_forward, 
            output_shape=(H, W), interpolation='cubic'
        )
        roundtrip_cubic = torch_projectors.forward_project_2d(
            proj_rotated_cubic.squeeze(1), rot_back_2d, 
            output_shape=(H, H), interpolation='cubic'
        )
        
        # Compare errors against reference
        error_linear = torch.mean(torch.abs(reference_proj - roundtrip_linear)**2).item()
        error_cubic = torch.mean(torch.abs(reference_proj - roundtrip_cubic)**2).item()
        
        linear_errors.append(error_linear)
        cubic_errors.append(error_cubic)
        
        print(f"  Linear round-trip MSE: {error_linear:.6f}")
        print(f"  Cubic round-trip MSE: {error_cubic:.6f}")
        print(f"  Improvement ratio: {error_linear / error_cubic:.2f}x")
    
    # Overall results
    avg_linear_error = sum(linear_errors) / len(linear_errors)
    avg_cubic_error = sum(cubic_errors) / len(cubic_errors)
    
    print(f"\nOverall Results:")
    print(f"Average linear round-trip MSE: {avg_linear_error:.6f}")
    print(f"Average cubic round-trip MSE: {avg_cubic_error:.6f}")
    print(f"Average improvement ratio: {avg_linear_error / avg_cubic_error:.2f}x")
    
    # Visualize results for Y-axis rotation (middle case)
    rot_forward_y = torch.tensor([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    rot_back_2d_y = torch.tensor([
        [cos_a, sin_a],
        [-sin_a, cos_a]
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    proj_y_linear = torch_projectors.forward_project_3d_to_2d(
        reconstruction_3d.unsqueeze(0), rot_forward_y, 
        output_shape=(H, W), interpolation='linear'
    )
    roundtrip_y_linear = torch_projectors.forward_project_2d(
        proj_y_linear.squeeze(1), rot_back_2d_y, 
        output_shape=(H, H), interpolation='linear'
    )
    
    proj_y_cubic = torch_projectors.forward_project_3d_to_2d(
        reconstruction_3d.unsqueeze(0), rot_forward_y, 
        output_shape=(H, W), interpolation='cubic'
    )
    roundtrip_y_cubic = torch_projectors.forward_project_2d(
        proj_y_cubic.squeeze(1), rot_back_2d_y, 
        output_shape=(H, H), interpolation='cubic'
    )
    
    # Visualize results
    plot_fourier_tensors(
        [reference_proj[0, 0].cpu(), roundtrip_y_linear[0, 0].cpu(), roundtrip_y_cubic[0, 0].cpu(), 
         (reference_proj[0, 0] - roundtrip_y_linear[0, 0]).cpu(), (reference_proj[0, 0] - roundtrip_y_cubic[0, 0]).cpu()],
        ['Reference (Identity)', 'Linear round-trip', 'Cubic round-trip', 'Linear error', 'Cubic error'],
        f'test_outputs/3d/interpolation_quality_comparison_3d_to_2d_{device.type}.png',
        shape=(1, 5)
    )
    
    # Verify cubic interpolation performs better on average
    assert avg_cubic_error < avg_linear_error, f"Cubic interpolation should be more accurate: cubic={avg_cubic_error:.6f} vs linear={avg_linear_error:.6f}"
    
    # Verify each axis shows improvement (cubic should be better for all axes)
    for i, (axis_name, _) in enumerate(test_angles):
        assert cubic_errors[i] < linear_errors[i], f"Cubic should be better than linear for {axis_name}: cubic={cubic_errors[i]:.6f} vs linear={linear_errors[i]:.6f}"