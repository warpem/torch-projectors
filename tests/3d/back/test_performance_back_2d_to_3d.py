"""
Performance and benchmarking tests for torch-projectors 2D->3D back-projection.

This module tests performance characteristics and compares interpolation quality
for 2D->3D back-projection operations.
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
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from test_utils import device, plot_fourier_tensors, create_fourier_mask, create_friedel_symmetric_noise

# Third-party benchmark import (optional)
try:
    from torch_fourier_slice import insert_central_slices_rfft_3d
    HAS_TORCH_FOURIER_SLICE = True
except ImportError:
    HAS_TORCH_FOURIER_SLICE = False


def test_backproject_2d_to_3d_performance_benchmark(device):
    """
    Performance benchmark comparing linear vs cubic interpolation for 2D->3D back-projection.
    
    Tests back-projection performance with:
    - Multiple projection sets (batch processing)
    - Random 3D rotation angles and shifts
    - Proper benchmarking practices (warmup, multiple runs, statistics)
    """
    
    # Benchmark parameters
    num_projection_sets = 8
    num_projections_per_set = 512
    H, W = 128, 65  # Larger size for more realistic performance test
    D = H  # Cubic reconstruction volume
    num_warmup_runs = 3
    num_timing_runs = 10
    
    print(f"\n2D->3D Back-projection Performance Benchmark:")
    print(f"- Projection sets: {num_projection_sets}")
    print(f"- Projections per set: {num_projections_per_set}")
    print(f"- Total projections: {num_projection_sets * num_projections_per_set}")
    print(f"- Image size: {H}x{H} -> {D}x{H}x{W}")
    print(f"- Warmup runs: {num_warmup_runs}, Timing runs: {num_timing_runs}")
    
    # Generate test data
    torch.manual_seed(42)
    projections = torch.randn(num_projection_sets, num_projections_per_set, H, W, dtype=torch.complex64, device=device)
    weights = torch.rand(num_projection_sets, num_projections_per_set, H, W, dtype=torch.float32, device=device)
    
    # Random 3D rotations (Z-axis rotations for simplicity in benchmark)
    angles = torch.rand(num_projection_sets, num_projections_per_set, device=device) * 2 * math.pi
    rotations = torch.zeros(num_projection_sets, num_projections_per_set, 3, 3, device=device)
    for i in range(num_projection_sets):
        for j in range(num_projections_per_set):
            angle = angles[i, j]
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            # 3x3 rotation matrix around Z-axis
            rotations[i, j] = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], device=device)
    
    shifts = torch.randn(num_projection_sets, num_projections_per_set, 2, device=device) * 10.0
    
    def benchmark_backproject_2d_to_3d_interpolation(interpolation_method):
        """Benchmark a specific interpolation method for 2D->3D back-projection"""
        forward_times = []
        backward_times = []
        
        # Warmup runs
        for _ in range(num_warmup_runs):
            projections.requires_grad_(True)
            weights.requires_grad_(True)
            reconstruction, weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                projections, rotations=rotations, weights=weights, shifts=shifts, 
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
            
            reconstruction, weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                projections, rotations=rotations, weights=weights, shifts=shifts,
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
    print(f"\nBenchmarking linear interpolation for 2D->3D back-projection...")
    linear_forward, linear_backward = benchmark_backproject_2d_to_3d_interpolation('linear')
    
    print(f"Benchmarking cubic interpolation for 2D->3D back-projection...")
    cubic_forward, cubic_backward = benchmark_backproject_2d_to_3d_interpolation('cubic')
    
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
    print(f"2D->3D BACK-PROJECTION PERFORMANCE RESULTS")
    print(f"{'='*60}")
    
    print(f"\nForward 2D->3D Back-projection Times (seconds):")
    print(f"Linear  - Mean: {linear_forward_stats['mean']:.4f} ± {linear_forward_stats['stdev']:.4f}")
    print(f"Cubic   - Mean: {cubic_forward_stats['mean']:.4f} ± {cubic_forward_stats['stdev']:.4f}")
    print(f"Slowdown: {cubic_forward_stats['mean'] / linear_forward_stats['mean']:.2f}x")
    
    print(f"\nBackward 2D->3D Back-projection Times (seconds):")
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


def test_backproject_2d_to_3d_interpolation_quality_comparison(device):
    """
    Compare linear vs cubic interpolation quality for 2D->3D back-projection using round-trip operations.
    
    Test procedure:
    1. Start with a known 3D reconstruction in Fourier space
    2. Forward project it 3D->2D at +45° to create 2D projections  
    3. Back-project those 2D projections to 3D at +45° with linear interpolation
    4. Back-project those 2D projections to 3D at +45° with cubic interpolation  
    5. Compare both 3D results to original 3D reconstruction, verify cubic is better
    """
    torch.manual_seed(42)
    
    # Create a known 3D reconstruction in Fourier space (RFFT format)
    # Generate on CPU first to ensure consistent data across devices
    D, H, W = 64, 64, 33  # 64x64x64 real -> 64x64x33 complex after RFFT
    
    # Create Friedel-symmetric noise (proper for real-valued 3D reconstruction)
    original_reconstruction = create_friedel_symmetric_noise((D, H, W), device='cpu').unsqueeze(0).to(device)
    
    # Apply circular mask in Fourier space to avoid edge artifacts
    max_radius = H // 2  # Leave some margin to avoid edge effects
    mask = create_fourier_mask((H, W), max_radius**2, device=device)
    for d in range(D):
        original_reconstruction[0, d][mask] = 0.0

    # Test rotation: +45° around Z-axis
    angle_rad = math.radians(45.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # 3D rotation matrix (+45° around Z-axis)
    rot_3d = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0], 
        [0, 0, 1]
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Step 1: Forward project the known 3D reconstruction at +45° with both interpolation methods
    projections_linear = torch_projectors.project_3d_to_2d_forw(
        original_reconstruction, rotations=rot_3d, output_shape=(H, H), interpolation='linear'
    )
    
    projections_cubic = torch_projectors.project_3d_to_2d_forw(
        original_reconstruction, rotations=rot_3d, output_shape=(H, H), interpolation='cubic'
    )
    
    # Step 2: Back-project with linear interpolation (using linear-generated projections)
    reconstructed_linear, _ = torch_projectors.backproject_2d_to_3d_forw(
        projections_linear, rotations=rot_3d, interpolation='linear'
    )
    
    # Step 3: Back-project with cubic interpolation (using cubic-generated projections)
    reconstructed_cubic, _ = torch_projectors.backproject_2d_to_3d_forw(
        projections_cubic, rotations=rot_3d, interpolation='cubic'
    )
    
    # Compare errors
    error_linear = torch.mean(torch.abs(original_reconstruction[0, 0] - reconstructed_linear[0, 0])**2).item()
    error_cubic = torch.mean(torch.abs(original_reconstruction[0, 0] - reconstructed_cubic[0, 0])**2).item()

    print(f"Linear 2D->3D back-projection MSE: {error_linear:.6f}")
    print(f"Cubic 2D->3D back-projection MSE: {error_cubic:.6f}")
    #print(f"Improvement ratio: {error_linear / error_cubic:.2f}x")
    
    # Visualize results (show central slices for 3D data)
    plot_fourier_tensors(
        [original_reconstruction[0, 0].cpu(), 
         projections_linear[0, 0].cpu(), 
         projections_cubic[0, 0].cpu(),
         reconstructed_linear[0, 0].cpu(),
         (original_reconstruction[0, 0] - reconstructed_linear[0, 0]).cpu(),
         reconstructed_cubic[0, 0].cpu(),
         (original_reconstruction[0, 0] - reconstructed_cubic[0, 0]).cpu()],
        ['Original 3D (z=0)', 'Linear Forward Proj', 'Cubic Forward Proj', 'Linear Back-proj (z=0)', 'Linear Error', 'Cubic Back-proj (z=0)', 'Cubic Error'],
        f'test_outputs/3d/back/backproject_2d_to_3d_interpolation_quality_comparison_{device.type}.png',
        shape=(2, 4)
    )
    
    # Verify cubic interpolation performs better
    assert error_cubic < error_linear, f"Cubic interpolation should be more accurate: cubic={error_cubic:.6f} vs linear={error_linear:.6f}"


def test_backproject_2d_to_3d_accumulation_performance(device):
    """
    Tests the performance characteristics of 2D->3D back-projection accumulation.
    Compares single large batch vs multiple smaller batches.
    """
    torch.manual_seed(42)
    
    # Test parameters
    total_projections = 1000
    H, W = 64, 33
    D = H  # Cubic reconstruction volume
    num_timing_runs = 5
    
    print(f"\n2D->3D Back-projection Accumulation Performance Test:")
    print(f"- Total projections: {total_projections}")
    print(f"- Image size: {H}x{(W-1)*2} -> {D}x{H}x{(W-1)*2}")
    print(f"- Timing runs: {num_timing_runs}")
    
    # Generate test data
    all_projections = torch.randn(total_projections, H, W, dtype=torch.complex64, device=device)
    angles = torch.rand(total_projections, device=device) * 2 * math.pi
    rotations = torch.zeros(total_projections, 3, 3, device=device)
    for i in range(total_projections):
        angle = angles[i]
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        # 3x3 rotation matrix around Z-axis
        rotations[i] = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], device=device)
    
    def time_backproject_2d_to_3d_batch(batch_size, interpolation='linear'):
        """Time 2D->3D back-projection with a specific batch size"""
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
                
                reconstruction, _ = torch_projectors.backproject_2d_to_3d_forw(
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
            mean_time, std_time = time_backproject_2d_to_3d_batch(batch_size)
            throughput = total_projections / mean_time
            print(f"{batch_size:10d} | {mean_time:12.4f} | {std_time:10.4f} | {throughput:15.1f}")
    
    # Compare interpolation methods for optimal batch size
    optimal_batch_size = 100  # Usually a good middle ground
    
    print(f"\nInterpolation Comparison (Batch Size = {optimal_batch_size}):")
    
    linear_time, linear_std = time_backproject_2d_to_3d_batch(optimal_batch_size, 'linear')
    cubic_time, cubic_std = time_backproject_2d_to_3d_batch(optimal_batch_size, 'cubic')
    
    print(f"Linear  - Mean: {linear_time:.4f} ± {linear_std:.4f} seconds, Throughput: {total_projections/linear_time:.1f} proj/s")
    print(f"Cubic   - Mean: {cubic_time:.4f} ± {cubic_std:.4f} seconds, Throughput: {total_projections/cubic_time:.1f} proj/s")
    print(f"Slowdown: {cubic_time / linear_time:.2f}x")
    
    # Verify that batching doesn't affect correctness
    # Back-project all at once
    all_projections_batched = all_projections.unsqueeze(0)  # Shape: (1, total_projections, H, W)
    all_rotations_batched = rotations.unsqueeze(0)  # Shape: (1, total_projections, 3, 3)
    
    reconstruction_all, _ = torch_projectors.backproject_2d_to_3d_forw(
        all_projections_batched, rotations=all_rotations_batched, interpolation='linear'
    )
    
    # Back-project in smaller batches and accumulate
    reconstruction_batched = torch.zeros_like(reconstruction_all[0])
    batch_size = 100
    
    for start_idx in range(0, total_projections, batch_size):
        end_idx = min(start_idx + batch_size, total_projections)
        batch_projections = all_projections[start_idx:end_idx].unsqueeze(0)
        batch_rotations = rotations[start_idx:end_idx].unsqueeze(0)
        
        batch_reconstruction, _ = torch_projectors.backproject_2d_to_3d_forw(
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
    assert max_diff < 5e-4, f"Batching affects results: max diff {max_diff:.2e}"
    assert mean_diff < 5e-5, f"Batching affects results: mean diff {mean_diff:.2e}"
    
    print("✅ Batching preserves correctness within numerical precision")


@pytest.mark.skipif(not HAS_TORCH_FOURIER_SLICE, reason="torch-fourier-slice not available")
def test_benchmark_torch_fourier_slice_2d_to_3d(device):
    """
    Benchmark torch-fourier-slice against torch-projectors 2D->3D back-projection.
    
    Uses same parameters as our benchmarks to enable direct comparison.
    Tests forward passes with proper timing methodology.
    """

    # Skip MPS - torch-fourier-slice operators not implemented for MPS
    if device.type == "mps":
        pytest.skip("torch-fourier-slice operators not implemented for MPS")

    torch.manual_seed(42)
    
    # Use same benchmark parameters as our performance test
    num_projection_sets = 1
    num_projections_per_set = 8192
    D, H, W = 128, 128, 128
    W_half = W // 2 + 1
    num_warmup_runs = 3
    num_timing_runs = 10
    
    print(f"\n2D->3D Third-Party Benchmark (torch-fourier-slice vs torch-projectors):")
    print(f"- Projection sets: {num_projection_sets}")
    print(f"- Projections per set: {num_projections_per_set}")
    print(f"- Total projections: {num_projection_sets * num_projections_per_set}")
    print(f"- Image size: {H}x{W} -> {D}x{H}x{W}")
    print(f"- Warmup runs: {num_warmup_runs}, Timing runs: {num_timing_runs}")
    
    # Generate test data - 2D projections for back-projection to 3D
    torch.manual_seed(42)
    
    # 2D projections in Fourier domain
    projections_tp = torch.randn(num_projection_sets, num_projections_per_set, H, W_half, dtype=torch.complex64, device=device)
    weights_tp = torch.randn(num_projection_sets, num_projections_per_set, H, W_half, dtype=torch.float32, device=device)
    projections_tfs = projections_tp.clone()  # torch-fourier-slice copy on same device
    
    # Generate rotation matrices using same method as our benchmark
    angles_x = torch.rand(num_projection_sets, num_projections_per_set) * 2 * math.pi
    angles_y = torch.rand(num_projection_sets, num_projections_per_set) * 2 * math.pi
    angles_z = torch.rand(num_projection_sets, num_projections_per_set) * 2 * math.pi
    
    # Create rotation matrices for torch-projectors (batch format)
    rotations_tp = torch.zeros(num_projection_sets, num_projections_per_set, 3, 3, device=device)
    
    # Create rotation matrices for torch-fourier-slice (per volume format)
    rotations_tfs = torch.zeros(num_projection_sets, num_projections_per_set, 3, 3, device=device)
    
    for i in range(num_projection_sets):
        for j in range(num_projections_per_set):
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
            rotation = Rz @ Ry @ Rx
            rotations_tp[i, j] = rotation
            rotations_tfs[i, j] = rotation
    
    shifts_tp = torch.randn(num_projection_sets, num_projections_per_set, 2, device=device) * 5.0
    
    def benchmark_torch_fourier_slice():
        """Benchmark torch-fourier-slice using 2D->3D back-projection"""
        forward_times = []
        backward_times = []
        
        # Warmup runs
        for _ in range(num_warmup_runs):
            projections_tfs.requires_grad_(True)
            all_volumes = []
            for i in range(num_projection_sets):
                projections = projections_tfs[i]  # Projections for this set (num_projections_per_set, H, W_half)
                rotations = rotations_tfs[i]  # Rotations for this set (num_projections_per_set, 3, 3)
                # Use 2D->3D back-projection API
                volume, weight_volume = insert_central_slices_rfft_3d(
                    projections, (D, H, W), rotations
                )
                all_volumes.append(volume)
            all_volumes = torch.stack(all_volumes)  # (num_projection_sets, D, H, W_half)
            loss = torch.sum(torch.abs(all_volumes)**2)
            loss.backward()
            projections_tfs.grad = None
        
        # Timing runs
        for _ in range(num_timing_runs):
            projections_tfs.requires_grad_(True)
            
            # Time forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            start_time = time.perf_counter()
            
            all_volumes = []
            for i in range(num_projection_sets):
                projections = projections_tfs[i]  # Projections for this set (num_projections_per_set, H, W_half)
                rotations = rotations_tfs[i]  # Rotations for this set (num_projections_per_set, 3, 3)
                # Use 2D->3D back-projection API
                volume, weight_volume = insert_central_slices_rfft_3d(
                    projections, (D, H, W), rotations
                )
                all_volumes.append(volume)
            
            # Stack all volumes
            all_volumes = torch.stack(all_volumes)  # (num_projection_sets, D, H, W_half)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            forward_time = time.perf_counter() - start_time
            forward_times.append(forward_time)
            
            # Time backward pass
            loss = torch.sum(torch.abs(all_volumes)**2)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            start_time = time.perf_counter()
            
            loss.backward()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            backward_time = time.perf_counter() - start_time
            backward_times.append(backward_time)
            
            projections_tfs.grad = None
        
        return forward_times, backward_times
    
    def benchmark_torch_projectors():
        """Benchmark torch-projectors"""
        forward_times = []
        backward_times = []
        
        # Warmup runs
        for _ in range(num_warmup_runs):
            projections_tp.requires_grad_(True)
            weights_tp.requires_grad_(True)
            reconstruction, weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                projections_tp, weights=weights_tp, rotations=rotations_tp, shifts=shifts_tp,
                interpolation='linear'
            )
            loss = torch.sum(torch.abs(reconstruction)**2) + 0.1 * torch.sum(weight_reconstruction**2)
            loss.backward()
            projections_tp.grad = None
            weights_tp.grad = None
        
        # Timing runs
        for _ in range(num_timing_runs):
            projections_tp.requires_grad_(True)
            weights_tp.requires_grad_(True)
            
            # Time forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            start_time = time.perf_counter()
            
            reconstruction, weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                projections_tp, weights=weights_tp, rotations=rotations_tp, shifts=shifts_tp,
                interpolation='linear'
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
            
            projections_tp.grad = None
            weights_tp.grad = None
        
        return forward_times, backward_times
    
    # Run benchmarks
    print(f"\nBenchmarking torch-fourier-slice...")
    tfs_forward, tfs_backward = benchmark_torch_fourier_slice()
    
    print(f"Benchmarking torch-projectors...")
    tp_forward, tp_backward = benchmark_torch_projectors()
    
    # Calculate statistics
    def calc_stats(times):
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times)
        }
    
    tfs_forward_stats = calc_stats(tfs_forward)
    tfs_backward_stats = calc_stats(tfs_backward)
    tp_forward_stats = calc_stats(tp_forward)
    tp_backward_stats = calc_stats(tp_backward)
    
    # Print comparison results
    print(f"\n{'='*80}")
    print(f"2D->3D LIBRARY COMPARISON RESULTS")
    print(f"{'='*80}")
    
    print(f"\nForward Back-projection Times (seconds):")
    print(f"torch-fourier-slice - Mean: {tfs_forward_stats['mean']:.4f} ± {tfs_forward_stats['stdev']:.4f}")
    print(f"torch-projectors   - Mean: {tp_forward_stats['mean']:.4f} ± {tp_forward_stats['stdev']:.4f}")
    
    if tp_forward_stats['mean'] > 0:
        speedup = tfs_forward_stats['mean'] / tp_forward_stats['mean']
        if speedup > 1:
            print(f"torch-projectors is {speedup:.2f}x faster")
        else:
            print(f"torch-fourier-slice is {1/speedup:.2f}x faster")
    
    print(f"\nBackward Back-projection Times (seconds):")
    print(f"torch-fourier-slice - Mean: {tfs_backward_stats['mean']:.4f} ± {tfs_backward_stats['stdev']:.4f}")
    print(f"torch-projectors   - Mean: {tp_backward_stats['mean']:.4f} ± {tp_backward_stats['stdev']:.4f}")
    
    if tp_backward_stats['mean'] > 0:
        speedup_back = tfs_backward_stats['mean'] / tp_backward_stats['mean']
        if speedup_back > 1:
            print(f"torch-projectors is {speedup_back:.2f}x faster")
        else:
            print(f"torch-fourier-slice is {1/speedup_back:.2f}x faster")
    
    print(f"\nTotal Times (Forward + Backward):")
    tfs_total = tfs_forward_stats['mean'] + tfs_backward_stats['mean']
    tp_total = tp_forward_stats['mean'] + tp_backward_stats['mean']
    print(f"torch-fourier-slice - {tfs_total:.4f} seconds")
    print(f"torch-projectors   - {tp_total:.4f} seconds")
    
    if tp_total > 0:
        total_speedup = tfs_total / tp_total
        if total_speedup > 1:
            print(f"torch-projectors is {total_speedup:.2f}x faster overall")
        else:
            print(f"torch-fourier-slice is {1/total_speedup:.2f}x faster overall")
    
    print(f"\nThroughput (back-projections/second):")
    total_projections = num_projection_sets * num_projections_per_set
    print(f"torch-fourier-slice - Forward: {total_projections / tfs_forward_stats['mean']:.1f}, Total: {total_projections / tfs_total:.1f}")
    print(f"torch-projectors   - Forward: {total_projections / tp_forward_stats['mean']:.1f}, Total: {total_projections / tp_total:.1f}")

    print(f"\nNote: Both torch-fourier-slice and torch-projectors operate on Fourier domain data.")
    print(f"This provides a fair comparison without FFT conversion overhead.")
    print(f"torch-projectors includes additional features like shifts and weight handling.")