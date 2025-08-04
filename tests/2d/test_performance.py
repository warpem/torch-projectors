"""
Performance and benchmarking tests for torch-projectors.

This module tests performance characteristics and compares interpolation quality.
"""

import torch
import torch_projectors
import pytest
import math
import time
import statistics
from test_utils import device, plot_fourier_tensors


def test_performance_benchmark(device):
    """
    Performance benchmark comparing linear vs cubic interpolation.
    
    Tests forward and backward projection performance with:
    - Multiple reconstructions (batch processing)
    - Random rotation angles and shifts
    - Proper benchmarking practices (warmup, multiple runs, statistics)
    """
    
    # Benchmark parameters
    num_reconstructions = 8
    num_projections_per_rec = 512
    H, W = 128, 65  # Larger size for more realistic performance test
    num_warmup_runs = 3
    num_timing_runs = 10
    
    print(f"\nPerformance Benchmark:")
    print(f"- Reconstructions: {num_reconstructions}")
    print(f"- Projections per reconstruction: {num_projections_per_rec}")
    print(f"- Total projections: {num_reconstructions * num_projections_per_rec}")
    print(f"- Image size: {H}x{H} -> {H}x{W}")
    print(f"- Warmup runs: {num_warmup_runs}, Timing runs: {num_timing_runs}")
    
    # Generate test data
    torch.manual_seed(42)
    reconstructions = torch.randn(num_reconstructions, H, W, dtype=torch.complex64, device=device)
    
    # Random rotations and shifts
    angles = torch.rand(num_reconstructions, num_projections_per_rec, device=device) * 2 * math.pi
    rotations = torch.zeros(num_reconstructions, num_projections_per_rec, 2, 2, device=device)
    for i in range(num_reconstructions):
        for j in range(num_projections_per_rec):
            angle = angles[i, j]
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rotations[i, j] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)
    
    shifts = torch.randn(num_reconstructions, num_projections_per_rec, 2, device=device) * 10.0
    
    def benchmark_interpolation(interpolation_method):
        """Benchmark a specific interpolation method"""
        forward_times = []
        backward_times = []
        
        # Warmup runs
        for _ in range(num_warmup_runs):
            reconstructions.requires_grad_(True)
            projections = torch_projectors.project_2d_forw(
                reconstructions, rotations, shifts, 
                output_shape=(H, H), interpolation=interpolation_method
            )
            loss = torch.sum(torch.abs(projections)**2)
            loss.backward()
            reconstructions.grad = None
        
        # Timing runs
        for run in range(num_timing_runs):
            reconstructions.requires_grad_(True)
            
            # Time forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            projections = torch_projectors.project_2d_forw(
                reconstructions, rotations, shifts,
                output_shape=(H, H), interpolation=interpolation_method
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
    print(f"PERFORMANCE RESULTS")
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


def test_interpolation_quality_comparison(device):
    """
    Compare linear vs cubic interpolation quality using round-trip projection.
    
    Test procedure:
    1. Start with random reconstruction
    2. Make identity projection (0°) as reference
    3. Project at +5° then back at -5° with linear interpolation
    4. Project at +5° then back at -5° with cubic interpolation  
    5. Compare both results to reference, verify cubic is better
    """
    torch.manual_seed(42)
    
    # Create random reconstruction in Fourier space (RFFT format)
    # Generate on CPU first to ensure consistent data across devices
    H, W = 64, 33  # 64x64 real -> 64x33 complex after RFFT
    reconstruction = torch.randn(H, W, dtype=torch.complex64, device='cpu').to(device)
    
    # Identity projection (0 degrees) as reference
    identity_rotation = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 2, 2]
    reference_proj = torch_projectors.project_2d_forw(
        reconstruction.unsqueeze(0), 
        identity_rotation, 
        output_shape=(H, H),
        interpolation='linear'  # Doesn't matter for identity
    )
    
    # Test rotations: +5° then -5°
    angle_rad = math.radians(1.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    rot_plus5 = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    rot_minus5 = torch.tensor([[cos_a, sin_a], [-sin_a, cos_a]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # Round-trip with linear interpolation
    proj_5deg_linear = torch_projectors.project_2d_forw(
        reconstruction.unsqueeze(0), rot_plus5, output_shape=(H, H), interpolation='linear'
    )
    # proj_5deg_linear is [1, 1, 64, 64] but we need [1, 64, 33] for next projection
    roundtrip_linear = torch_projectors.project_2d_forw(
        proj_5deg_linear.squeeze(1), rot_minus5, output_shape=(H, H), interpolation='linear'
    )
    
    # Round-trip with cubic interpolation  
    proj_5deg_cubic = torch_projectors.project_2d_forw(
        reconstruction.unsqueeze(0), rot_plus5, output_shape=(H, H), interpolation='cubic'
    )
    # proj_5deg_cubic is [1, 1, 64, 64] but we need [1, 64, 33] for next projection
    roundtrip_cubic = torch_projectors.project_2d_forw(
        proj_5deg_cubic.squeeze(1), rot_minus5, output_shape=(H, H), interpolation='cubic'
    )
    
    # Compare errors
    error_linear = torch.mean(torch.abs(reference_proj - roundtrip_linear)**2).item()
    error_cubic = torch.mean(torch.abs(reference_proj - roundtrip_cubic)**2).item()
    
    print(f"Linear round-trip MSE: {error_linear:.6f}")
    print(f"Cubic round-trip MSE: {error_cubic:.6f}")
    print(f"Improvement ratio: {error_linear / error_cubic:.2f}x")
    
    # Visualize results
    plot_fourier_tensors(
        [reference_proj[0].cpu(), roundtrip_linear[0].cpu(), roundtrip_cubic[0].cpu(), 
         (reference_proj[0] - roundtrip_linear[0]).cpu(), (reference_proj[0] - roundtrip_cubic[0]).cpu()],
        ['Reference (0°)', 'Linear round-trip', 'Cubic round-trip', 'Linear error', 'Cubic error'],
        f'test_outputs/interpolation_quality_comparison_{device.type}.png',
        shape=(1, 5)
    )
    
    # Verify cubic interpolation performs better
    assert error_cubic < error_linear, f"Cubic interpolation should be more accurate: cubic={error_cubic:.6f} vs linear={error_linear:.6f}"