"""
Performance and benchmarking tests for torch-projectors back-projection.

This module tests performance characteristics and compares interpolation quality
for back-projection operations.
"""

import torch
import torch_projectors
import pytest
import math
import time
import statistics
import sys
import os

# Add parent directory to path to import test_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_utils import device, plot_fourier_tensors, create_fourier_mask, create_friedel_symmetric_noise


def test_backproject_performance_benchmark(device):
    """
    Performance benchmark comparing linear vs cubic interpolation for back-projection.
    
    Tests back-projection performance with:
    - Multiple projection sets (batch processing)
    - Random rotation angles and shifts
    - Proper benchmarking practices (warmup, multiple runs, statistics)
    """
    
    # Benchmark parameters
    num_projection_sets = 8
    num_projections_per_set = 512
    H, W = 128, 65  # Larger size for more realistic performance test
    num_warmup_runs = 3
    num_timing_runs = 10
    
    print(f"\nBack-projection Performance Benchmark:")
    print(f"- Projection sets: {num_projection_sets}")
    print(f"- Projections per set: {num_projections_per_set}")
    print(f"- Total projections: {num_projection_sets * num_projections_per_set}")
    print(f"- Image size: {H}x{H} -> {H}x{W}")
    print(f"- Warmup runs: {num_warmup_runs}, Timing runs: {num_timing_runs}")
    
    # Generate test data
    torch.manual_seed(42)
    projections = torch.randn(num_projection_sets, num_projections_per_set, H, W, dtype=torch.complex64, device=device)
    weights = torch.rand(num_projection_sets, num_projections_per_set, H, W, dtype=torch.float32, device=device)
    
    # Random rotations and shifts
    angles = torch.rand(num_projection_sets, num_projections_per_set, device=device) * 2 * math.pi
    rotations = torch.zeros(num_projection_sets, num_projections_per_set, 2, 2, device=device)
    for i in range(num_projection_sets):
        for j in range(num_projections_per_set):
            angle = angles[i, j]
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rotations[i, j] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)
    
    shifts = torch.randn(num_projection_sets, num_projections_per_set, 2, device=device) * 10.0
    
    def benchmark_backproject_interpolation(interpolation_method):
        """Benchmark a specific interpolation method for back-projection"""
        forward_times = []
        backward_times = []
        
        # Warmup runs
        for _ in range(num_warmup_runs):
            projections.requires_grad_(True)
            weights.requires_grad_(True)
            reconstruction, weight_reconstruction = torch_projectors.backproject_2d_forw(
                projections, rotations, weights=weights, shifts=shifts, 
                interpolation=interpolation_method
            )
            loss = torch.sum(torch.abs(reconstruction)**2) + 0.1 * torch.sum(weight_reconstruction**2)
            loss.backward()
            projections.grad = None
            weights.grad = None
        
        # Timing runs
        for run in range(num_timing_runs):
            projections.requires_grad_(True)
            weights.requires_grad_(True)
            
            # Time forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            reconstruction, weight_reconstruction = torch_projectors.backproject_2d_forw(
                projections, rotations, weights=weights, shifts=shifts,
                interpolation=interpolation_method
            )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            forward_time = time.perf_counter() - start_time
            forward_times.append(forward_time)
            
            # Time backward pass
            loss = torch.sum(torch.abs(reconstruction)**2) + 0.1 * torch.sum(weight_reconstruction**2)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            start_time = time.perf_counter()
            
            loss.backward()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            backward_time = time.perf_counter() - start_time
            backward_times.append(backward_time)
            
            projections.grad = None
            weights.grad = None
        
        return forward_times, backward_times
    
    # Benchmark both methods
    print(f"\nBenchmarking linear interpolation for back-projection...")
    linear_forward, linear_backward = benchmark_backproject_interpolation('linear')
    
    print(f"Benchmarking cubic interpolation for back-projection...")
    cubic_forward, cubic_backward = benchmark_backproject_interpolation('cubic')
    
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
    print(f"BACK-PROJECTION PERFORMANCE RESULTS")
    print(f"{'='*60}")
    
    print(f"\nForward Back-projection Times (seconds):")
    print(f"Linear  - Mean: {linear_forward_stats['mean']:.4f} ± {linear_forward_stats['stdev']:.4f}")
    print(f"Cubic   - Mean: {cubic_forward_stats['mean']:.4f} ± {cubic_forward_stats['stdev']:.4f}")
    print(f"Slowdown: {cubic_forward_stats['mean'] / linear_forward_stats['mean']:.2f}x")
    
    print(f"\nBackward Back-projection Times (seconds):")
    print(f"Linear  - Mean: {linear_backward_stats['mean']:.4f} ± {linear_backward_stats['stdev']:.4f}")
    print(f"Cubic   - Mean: {cubic_backward_stats['mean']:.4f} ± {cubic_backward_stats['stdev']:.4f}")
    print(f"Slowdown: {cubic_backward_stats['mean'] / linear_backward_stats['mean']:.2f}x")
    
    print(f"\nTotal Times (Forward + Backward):")
    linear_total = linear_forward_stats['mean'] + linear_backward_stats['mean']
    cubic_total = cubic_forward_stats['mean'] + cubic_backward_stats['mean']
    print(f"Linear  - {linear_total:.4f} seconds")
    print(f"Cubic   - {cubic_total:.4f} seconds")
    print(f"Slowdown: {cubic_total / linear_total:.2f}x")
    
    print(f"\nThroughput (back-projections/second):")
    total_backprojections = num_projection_sets * num_projections_per_set
    print(f"Linear  - {total_backprojections / linear_total:.1f}")
    print(f"Cubic   - {total_backprojections / cubic_total:.1f}")


def test_backproject_interpolation_quality_comparison(device):
    """
    Compare linear vs cubic interpolation quality for back-projection using round-trip operations.
    
    Test procedure:
    1. Start with a known reconstruction in Fourier space
    2. Forward project it at 90° to create projections
    3. Back-project those projections at -90° with linear interpolation
    4. Back-project those projections at -90° with cubic interpolation  
    5. Compare both results to original reconstruction, verify cubic is better
    """
    torch.manual_seed(42)
    
    # Create a known reconstruction in Fourier space (RFFT format)
    # Generate on CPU first to ensure consistent data across devices
    H, W = 64, 33  # 64x64 real -> 64x33 complex after RFFT
    
    # Create Friedel-symmetric noise (proper for real-valued reconstruction)
    original_reconstruction = create_friedel_symmetric_noise((H, W), device='cpu').unsqueeze(0).to(device)
    
    # Apply circular mask in Fourier space to avoid edge artifacts
    max_radius = H // 2  # Leave some margin to avoid edge effects
    mask = create_fourier_mask((H, W), max_radius**2, device=device)
    original_reconstruction[0][mask] = 0.0

    # Test rotation: +45°
    angle_rad = math.radians(45.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # Forward projection rotation (+45°)
    rot = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Step 1: Forward project the known reconstruction at +45° with both interpolation methods
    projections_linear = torch_projectors.project_2d_forw(
        original_reconstruction, rotations=rot, output_shape=(H, H), interpolation='linear'
    )
    
    projections_cubic = torch_projectors.project_2d_forw(
        original_reconstruction, rotations=rot, output_shape=(H, H), interpolation='cubic'
    )
    
    # Step 2: Back-project with linear interpolation (using linear-generated projections)
    reconstructed_linear, _ = torch_projectors.backproject_2d_forw(
        projections_linear, rotations=rot, interpolation='linear'
    )
    
    # Step 3: Back-project with cubic interpolation (using cubic-generated projections)
    reconstructed_cubic, _ = torch_projectors.backproject_2d_forw(
        projections_cubic, rotations=rot, interpolation='cubic'
    )
    
    # Compare errors
    error_linear = torch.mean(torch.abs(original_reconstruction[0] - reconstructed_linear[0])**2).item()
    error_cubic = torch.mean(torch.abs(original_reconstruction[0] - reconstructed_cubic[0])**2).item()
    
    print(f"Linear back-projection MSE: {error_linear:.6f}")
    print(f"Cubic back-projection MSE: {error_cubic:.6f}")
    #print(f"Improvement ratio: {error_linear / error_cubic:.2f}x")
    
    # Visualize results
    plot_fourier_tensors(
        [original_reconstruction[0].cpu(), 
         projections_linear[0].cpu(), 
         projections_cubic[0].cpu(),
         reconstructed_linear[0].cpu(),
         (original_reconstruction[0] - reconstructed_linear[0]).cpu(),
         reconstructed_cubic[0].cpu(),
         (original_reconstruction[0] - reconstructed_cubic[0]).cpu()],
        ['Original Reconstruction', 'Linear Forward Projection', 'Cubic Forward Projection', 'Linear Back-projection', 'Linear Error', 'Cubic Back-projection', 'Cubic Error'],
        f'test_outputs/2d/back/backproject_interpolation_quality_comparison_{device.type}.png',
        shape=(2, 4)
    )
    
    # Verify cubic interpolation performs better
    assert error_cubic < error_linear, f"Cubic interpolation should be more accurate: cubic={error_cubic:.6f} vs linear={error_linear:.6f}"


def test_backproject_accumulation_performance(device):
    """
    Tests the performance characteristics of back-projection accumulation.
    Compares single large batch vs multiple smaller batches.
    """
    torch.manual_seed(42)
    
    # Test parameters
    total_projections = 1000
    H, W = 64, 33
    num_timing_runs = 5
    
    print(f"\nBack-projection Accumulation Performance Test:")
    print(f"- Total projections: {total_projections}")
    print(f"- Image size: {H}x{(W-1)*2}")
    print(f"- Timing runs: {num_timing_runs}")
    
    # Generate test data
    all_projections = torch.randn(total_projections, H, W, dtype=torch.complex64, device=device)
    angles = torch.rand(total_projections, device=device) * 2 * math.pi
    rotations = torch.zeros(total_projections, 2, 2, device=device)
    for i in range(total_projections):
        angle = angles[i]
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rotations[i] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)
    
    def time_backproject_batch(batch_size, interpolation='linear'):
        """Time back-projection with a specific batch size"""
        times = []
        
        for run in range(num_timing_runs):
            total_time = 0.0
            
            for start_idx in range(0, total_projections, batch_size):
                end_idx = min(start_idx + batch_size, total_projections)
                batch_projections = all_projections[start_idx:end_idx].unsqueeze(0)  # Add batch dim
                batch_rotations = rotations[start_idx:end_idx].unsqueeze(0)  # Add batch dim
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch.mps.synchronize() if torch.backends.mps.is_available() else None
                start_time = time.perf_counter()
                
                reconstruction, _ = torch_projectors.backproject_2d_forw(
                    batch_projections, rotations=batch_rotations, interpolation=interpolation
                )
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch.mps.synchronize() if torch.backends.mps.is_available() else None
                total_time += time.perf_counter() - start_time
            
            times.append(total_time)
        
        return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0
    
    # Test different batch sizes
    batch_sizes = [1, 10, 50, 100, 500, 1000]
    
    print("\nBatch Size Performance (Linear Interpolation):")
    print("Batch Size | Mean Time (s) | Std Dev (s) | Throughput (proj/s)")
    print("-" * 65)
    
    for batch_size in batch_sizes:
        if batch_size <= total_projections:
            mean_time, std_time = time_backproject_batch(batch_size)
            throughput = total_projections / mean_time
            print(f"{batch_size:10d} | {mean_time:12.4f} | {std_time:10.4f} | {throughput:15.1f}")
    
    # Compare interpolation methods for optimal batch size
    optimal_batch_size = 100  # Usually a good middle ground
    
    print(f"\nInterpolation Comparison (Batch Size = {optimal_batch_size}):")
    
    linear_time, linear_std = time_backproject_batch(optimal_batch_size, 'linear')
    cubic_time, cubic_std = time_backproject_batch(optimal_batch_size, 'cubic')
    
    print(f"Linear  - Mean: {linear_time:.4f} ± {linear_std:.4f} seconds, Throughput: {total_projections/linear_time:.1f} proj/s")
    print(f"Cubic   - Mean: {cubic_time:.4f} ± {cubic_std:.4f} seconds, Throughput: {total_projections/cubic_time:.1f} proj/s")
    print(f"Slowdown: {cubic_time / linear_time:.2f}x")
    
    # Verify that batching doesn't affect correctness
    # Back-project all at once
    all_projections_batched = all_projections.unsqueeze(0)  # Shape: (1, total_projections, H, W)
    all_rotations_batched = rotations.unsqueeze(0)  # Shape: (1, total_projections, 2, 2)
    
    reconstruction_all, _ = torch_projectors.backproject_2d_forw(
        all_projections_batched, rotations=all_rotations_batched, interpolation='linear'
    )
    
    # Back-project in smaller batches and accumulate
    reconstruction_batched = torch.zeros_like(reconstruction_all[0])
    batch_size = 100
    
    for start_idx in range(0, total_projections, batch_size):
        end_idx = min(start_idx + batch_size, total_projections)
        batch_projections = all_projections[start_idx:end_idx].unsqueeze(0)
        batch_rotations = rotations[start_idx:end_idx].unsqueeze(0)
        
        batch_reconstruction, _ = torch_projectors.backproject_2d_forw(
            batch_projections, rotations=batch_rotations, interpolation='linear'
        )
        reconstruction_batched += batch_reconstruction[0]
    
    # Compare results
    diff = torch.abs(reconstruction_all[0] - reconstruction_batched)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"\nBatching Correctness Check:")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    # Results should be identical within numerical precision
    assert max_diff < 1e-5, f"Batching affects results: max diff {max_diff:.2e}"
    assert mean_diff < 1e-6, f"Batching affects results: mean diff {mean_diff:.2e}"
    
    print("✅ Batching preserves correctness within numerical precision")