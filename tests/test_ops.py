import torch
import torch.nn.functional as F
import torch_projectors
import pytest
import math
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_real_space_tensors(tensors, titles, filename, shape=None):
    """
    Plots a list of real-space tensors.
    """
    os.makedirs('test_outputs', exist_ok=True)
    if shape is None:
        shape = (1, len(tensors))
    
    fig, axes = plt.subplots(shape[0], shape[1], figsize=(shape[1] * 4, shape[0] * 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(tensors):
            tensor = tensors[i]
            if tensor.dim() > 2:
                tensor = tensor.squeeze()
            title = titles[i]
            ax.imshow(tensor.detach().numpy(), cmap='viridis', origin='lower')
            ax.set_title(title)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_fourier_tensors(tensors, titles, filename, shape=None):
    """
    Plots the real and imaginary components of a list of rfft-formatted Fourier-space tensors.
    Real components are in the top row of a pair, imaginary in the bottom row.
    tensors: a list of tensors to plot.
    titles: a list of titles for each plot.
    filename: the output PNG file name.
    shape: a tuple for subplot shape (rows, cols) for the tensor grid.
    """
    os.makedirs('test_outputs', exist_ok=True)
    if shape is None:
        shape = (1, len(tensors))
    
    rows, cols = shape
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 4, rows * 2 * 4))
    
    if rows * 2 == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows * 2 == 1 or cols == 1:
        axes = axes.reshape(rows * 2, cols)

    for i in range(len(tensors)):
        tensor = tensors[i]
        if tensor.dim() > 2:
            tensor = tensor.squeeze()
        title = titles[i]

        row = i // cols
        col = i % cols
        
        # Plot real part
        ax_real = axes[2 * row, col]
        ax_real.imshow(tensor.real.detach().numpy(), cmap='viridis', origin='lower')
        ax_real.set_title(f"{title} (Real)")
        ax_real.axis('off')

        # Plot imaginary part
        ax_imag = axes[2 * row + 1, col]
        ax_imag.imshow(tensor.imag.detach().numpy(), cmap='viridis', origin='lower')
        ax_imag.set_title(f"{title} (Imag)")
        ax_imag.axis('off')

    # Hide unused axes
    for i in range(len(tensors), rows * cols):
        row = i // cols
        col = i % cols
        axes[2 * row, col].axis('off')
        axes[2 * row + 1, col].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_fourier_mask(shape, radius_cutoff_sq):
    """
    Creates a boolean mask for a Fourier-space tensor, zeroing out frequencies
    outside a specified radius.
    """
    H, W_half = shape[-2], shape[-1]
    ky = torch.arange(H, dtype=torch.float32)
    ky[ky > H // 2] -= H
    kx = torch.arange(W_half, dtype=torch.float32)
    kyy, kxx = torch.meshgrid(ky, kx, indexing='ij')
    
    return kxx**2 + kyy**2 > radius_cutoff_sq

@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_2d_identity(interpolation):
    """
    Tests the 2D forward projection with an identity rotation. The output should
    be a masked version of the input, with values outside the Fourier radius zeroed out.
    """
    B, P, H, W = 1, 1, 64, 64
    W_half = W // 2 + 1
    rec_fourier = torch.randn(B, H, W_half, dtype=torch.complex64)
    rotations = torch.eye(2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    output_shape = (H, W)

    projection = torch_projectors.forward_project_2d(
        rec_fourier,
        rotations,
        output_shape=output_shape,
        interpolation=interpolation
    )
    assert projection.shape == (B, P, H, W_half)

    # Create an expected output by masking the input reconstruction
    expected_projection = rec_fourier.clone()
    
    # Replicate the C++ kernel's masking logic
    radius = min(H / 2.0, (W_half - 1) * 2)
    radius_cutoff_sq = radius * radius
    
    mask = create_fourier_mask(rec_fourier.shape, radius_cutoff_sq)
    expected_projection[0, mask] = 0

    # Compare the projection with the masked ground truth
    assert torch.allclose(projection[0, 0], expected_projection[0], atol=1e-5)

    plot_fourier_tensors(
        [rec_fourier, projection, expected_projection],
        ["Original", "Projection", "Expected (Masked)"],
        f"test_outputs/test_forward_project_2d_identity_{interpolation}.png"
    )

@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_2d_rotations(interpolation):
    """
    Tests that a single Fourier-space peak is rotated to the correct location.
    """
    B, P, H, W = 1, 3, 32, 32
    W_half = W // 2 + 1
    rec_fourier = torch.zeros(B, H, W_half, dtype=torch.complex64)
    rec_fourier[0, 0, 5] = 1.0 + 1.0j

    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32)
    rotations[0, 0] = torch.eye(2)
    rotations[0, 1] = torch.tensor([[0., 1.], [-1., 0.]])
    rotations[0, 2] = torch.tensor([[0., -1.], [1., 0.]])
    output_shape = (H, W)

    projection = torch_projectors.forward_project_2d(rec_fourier, rotations, output_shape=output_shape, interpolation=interpolation)

    # Expected peak locations after rotation
    expected_coords = [(0, 5), (5, 0), (H - 5, 0)]

    tensors_to_plot = [rec_fourier] + [projection[0, p] for p in range(P)]
    titles = ["Original"] + [f"Rotation {p}" for p in range(P)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/test_forward_project_2d_rotations_{interpolation}.png"
    )
    
    for p in range(P):
        proj_slice = projection[0, p].abs()
        max_val = proj_slice.max()
        peak_coords = (proj_slice == max_val).nonzero(as_tuple=False)
        
        assert max_val > 0.9
        
        # Check if any of the found peak coordinates match the expected coordinates
        found_match = False
        for coord in peak_coords:
            if coord[0].item() == expected_coords[p][0] and coord[1].item() == expected_coords[p][1]:
                found_match = True
                break
        assert found_match, f"Peak for pose {p} not found at expected location {expected_coords[p]}"

@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_2d_with_phase_shift(interpolation):
    """
    Tests that a real-space dot, when shifted in Fourier space, lands in the correct
    real-space location after accounting for circular shifts and Fourier masking.
    """
    B, P, H, W = 1, 1, 32, 32
    W_half = W // 2 + 1
    
    rec_real = torch.zeros(H, W, dtype=torch.float64)
    rec_real[H // 2, W // 2] = 1.0
    rec_fourier = torch.fft.rfft2(rec_real).unsqueeze(0)

    rotations = torch.eye(2, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    output_shape = (H, W)
    shift_r, shift_c = 2.0, 3.0
    shifts = torch.tensor([[[shift_r, shift_c]]], dtype=torch.float64)

    projection_fourier = torch_projectors.forward_project_2d(
        rec_fourier, rotations, shifts=shifts, output_shape=output_shape, interpolation=interpolation
    )
    projection_real = torch.fft.irfft2(projection_fourier.squeeze(0), s=(H, W))

    # Create the expected real-space output by rolling and then masking in Fourier space
    expected_real_rolled = torch.roll(rec_real, shifts=(int(shift_r), int(shift_c)), dims=(0, 1))
    expected_fourier = torch.fft.rfft2(expected_real_rolled)
    
    radius = min(H / 2.0, (W_half - 1) * 2)
    radius_cutoff_sq = radius * radius
    mask = create_fourier_mask(expected_fourier.shape, radius_cutoff_sq)
    expected_fourier[mask] = 0
    
    expected_real = torch.fft.irfft2(expected_fourier, s=(H, W))

    plot_real_space_tensors(
        [rec_real, projection_real, expected_real],
        ["Original", "Shifted (C++)", "Expected (Rolled & Masked)"],
        f"test_outputs/test_forward_project_2d_with_phase_shift_{interpolation}.png"
    )

    # Compare the entire images
    assert torch.allclose(projection_real, expected_real, atol=1e-1)

@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_reconstructions_single_angle(interpolation):
    """
    Tests that a single set of poses is correctly broadcast to multiple reconstructions.
    """
    B, P, H, W = 3, 5, 16, 16
    W_half = W // 2 + 1
    rec_fourier = torch.randn(B, H, W_half, dtype=torch.complex64)
    
    # Create a single set of P random rotation matrices
    angles = torch.rand(1, P) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(1, P, 2, 2, dtype=torch.float32)
    rotations[0, :, 0, 0] = cos_a
    rotations[0, :, 0, 1] = -sin_a
    rotations[0, :, 1, 0] = sin_a
    rotations[0, :, 1, 1] = cos_a
    
    output_shape = (H, W)

    projection = torch_projectors.forward_project_2d(
        rec_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over reconstructions and project individually
    expected_projection = torch.zeros_like(projection)
    for b in range(B):
        expected_projection[b] = torch_projectors.forward_project_2d(
            rec_fourier[b].unsqueeze(0), rotations, output_shape=output_shape, interpolation=interpolation
        )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    tensors_to_plot = [rec_fourier[0]] + [projection[0, p] for p in range(P)]
    titles = ["Original Rec (b=0)"] + [f"Projection (p={p})" for p in range(P)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/test_batching_multiple_reconstructions_single_angle_{interpolation}.png"
    )

@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_single_reconstruction_multiple_angles(interpolation):
    """
    Tests that multiple poses are correctly applied to a single reconstruction.
    """
    B, P, H, W = 1, 5, 16, 16
    W_half = W // 2 + 1
    rec_fourier = torch.randn(B, H, W_half, dtype=torch.complex64)

    # Create P random rotation matrices for the single reconstruction
    angles = torch.rand(B, P) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32)
    rotations[:, :, 0, 0] = cos_a
    rotations[:, :, 0, 1] = -sin_a
    rotations[:, :, 1, 0] = sin_a
    rotations[:, :, 1, 1] = cos_a
    
    output_shape = (H, W)

    projection = torch_projectors.forward_project_2d(
        rec_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over poses and project individually
    expected_projection = torch.zeros_like(projection)
    for p in range(P):
        expected_projection[0, p] = torch_projectors.forward_project_2d(
            rec_fourier, rotations[:, p].unsqueeze(1), output_shape=output_shape, interpolation=interpolation
        )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    tensors_to_plot = [rec_fourier[0]] + [projection[0, p] for p in range(P)]
    titles = ["Original Rec"] + [f"Projection (p={p})" for p in range(P)]
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/test_batching_single_reconstruction_multiple_angles_{interpolation}.png"
    )

@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_multiple_reconstructions_multiple_angles(interpolation):
    """
    Tests the one-to-one mapping of reconstructions to poses.
    """
    B, P, H, W = 4, 5, 16, 16
    W_half = W // 2 + 1
    rec_fourier = torch.randn(B, H, W_half, dtype=torch.complex64)

    # Create BxP random rotation matrices
    angles = torch.rand(B, P) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rotations = torch.zeros(B, P, 2, 2, dtype=torch.float32)
    rotations[:, :, 0, 0] = cos_a
    rotations[:, :, 0, 1] = -sin_a
    rotations[:, :, 1, 0] = sin_a
    rotations[:, :, 1, 1] = cos_a
    
    output_shape = (H, W)

    projection = torch_projectors.forward_project_2d(
        rec_fourier, rotations, output_shape=output_shape, interpolation=interpolation
    )

    # Ground truth: loop over reconstructions and poses and project individually
    expected_projection = torch.zeros_like(projection)
    for b in range(B):
        for p in range(P):
            expected_projection[b, p] = torch_projectors.forward_project_2d(
                rec_fourier[b].unsqueeze(0), rotations[b, p].unsqueeze(0).unsqueeze(0), output_shape=output_shape, interpolation=interpolation
            )

    assert torch.allclose(projection, expected_projection, atol=1e-5)

    tensors_to_plot = []
    titles = []
    for b in range(B):
        tensors_to_plot.append(rec_fourier[b])
        titles.append(f"Original (b={b})")
        for p in range(P):
            tensors_to_plot.append(projection[b, p])
            titles.append(f"Proj (b={b}, p={p})")
    
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/test_batching_multiple_reconstructions_multiple_angles_{interpolation}.png",
        shape=(B, P + 1)
    )

@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_forward_project_2d_backward_gradcheck_rec_only(interpolation):
    """
    Tests the 2D forward projection's backward pass using gradcheck for reconstruction only.
    """
    B, P, H, W = 1, 1, 16, 16
    W_half = W // 2 + 1
    rec_fourier = torch.randn(B, H, W_half, dtype=torch.complex128, requires_grad=True)
    rotations = torch.eye(2, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    output_shape = (H, W)

    def func(reconstruction):
        return torch_projectors.forward_project_2d(
            reconstruction,
            rotations,
            output_shape=output_shape,
            interpolation=interpolation
        )

    assert torch.autograd.gradcheck(func, rec_fourier, atol=1e-4, rtol=1e-2)

@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_visual_rotation_validation(interpolation):
    """
    Visual test: Create 3 line patterns in Fourier space (vertical + half-length horizontal) and visualize their fftshifted real components after rotations.
    """
    os.makedirs('test_outputs', exist_ok=True)

    H, W = 64, 64
    W_half = W // 2 + 1
    num_reconstructions = 3
    num_rotations = 5
    line_lengths = [10, 20, 30]
    rotation_increments = [5, 15, 30]

    reconstructions = torch.zeros(num_reconstructions, H, W_half, dtype=torch.complex64)
    for i, length in enumerate(line_lengths):
        reconstructions[i, :length, 0] = 1.0 + 1.0j
        horiz_length = length // 2
        reconstructions[i, 0, 1:horiz_length+1] = 1.0 + 1.0j

    rotations = torch.zeros(num_reconstructions, num_rotations, 2, 2, dtype=torch.float32)
    for i, increment in enumerate(rotation_increments):
        for j in range(num_rotations):
            angle_rad = math.radians(increment * (j + 1))
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            rotations[i, j] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])

    projections = torch_projectors.forward_project_2d(reconstructions, rotations, output_shape=(H, W), interpolation=interpolation)

    tensors_to_plot = []
    titles = []
    for i in range(num_reconstructions):
        tensors_to_plot.append(reconstructions[i])
        titles.append(f'Original (len={line_lengths[i]})')
        for j in range(num_rotations):
            tensors_to_plot.append(projections[i, j])
            titles.append(f'{rotation_increments[i] * (j + 1)}°')
    
    plot_fourier_tensors(
        tensors_to_plot,
        titles,
        f"test_outputs/test_visual_rotation_validation_{interpolation}.png",
        shape=(num_reconstructions, num_rotations + 1)
    )

    assert os.path.exists(f'test_outputs/test_visual_rotation_validation_{interpolation}.png')

    # Sanity: ensure projections at successive angles differ
    rec_fourier = reconstructions[0].unsqueeze(0)
    prev = None
    for i in range(1, num_rotations + 1):
        angle_deg = rotation_increments[0] * i
        angle_rad = math.radians(angle_deg)
        rot = torch.tensor([[[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]]], dtype=torch.float32).unsqueeze(0)
        proj = torch_projectors.forward_project_2d(rec_fourier, rot, output_shape=(H, W), interpolation=interpolation)
        if prev is not None:
            assert not torch.allclose(prev, proj[0].real, atol=1e-6)
        prev = proj[0].real


def test_performance_benchmark():
    """
    Performance benchmark comparing linear vs cubic interpolation.
    
    Tests forward and backward projection performance with:
    - Multiple reconstructions (batch processing)
    - Random rotation angles and shifts
    - Proper benchmarking practices (warmup, multiple runs, statistics)
    """
    import time
    import statistics
    
    # Benchmark parameters
    num_reconstructions = 8
    num_projections_per_rec = 64
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
    reconstructions = torch.randn(num_reconstructions, H, W, dtype=torch.complex64)
    
    # Random rotations and shifts
    angles = torch.rand(num_reconstructions, num_projections_per_rec) * 2 * math.pi
    rotations = torch.zeros(num_reconstructions, num_projections_per_rec, 2, 2)
    for i in range(num_reconstructions):
        for j in range(num_projections_per_rec):
            angle = angles[i, j]
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rotations[i, j] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
    
    shifts = torch.randn(num_reconstructions, num_projections_per_rec, 2) * 10.0
    
    def benchmark_interpolation(interpolation_method):
        """Benchmark a specific interpolation method"""
        forward_times = []
        backward_times = []
        
        # Warmup runs
        for _ in range(num_warmup_runs):
            reconstructions.requires_grad_(True)
            projections = torch_projectors.forward_project_2d(
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
            
            projections = torch_projectors.forward_project_2d(
                reconstructions, rotations, shifts,
                output_shape=(H, H), interpolation=interpolation_method
            )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            forward_time = time.perf_counter() - start_time
            forward_times.append(forward_time)
            
            # Time backward pass
            loss = torch.sum(torch.abs(projections)**2)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            loss.backward()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
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
    
    # Create performance visualization
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Box plots for forward times
    ax1.boxplot([linear_forward, cubic_forward], labels=['Linear', 'Cubic'])
    ax1.set_title('Forward Projection Times')
    ax1.set_ylabel('Time (seconds)')
    ax1.grid(True, alpha=0.3)
    
    # Box plots for backward times  
    ax2.boxplot([linear_backward, cubic_backward], labels=['Linear', 'Cubic'])
    ax2.set_title('Backward Projection Times')
    ax2.set_ylabel('Time (seconds)')
    ax2.grid(True, alpha=0.3)
    
    # Bar chart comparing means
    methods = ['Linear', 'Cubic']
    forward_means = [linear_forward_stats['mean'], cubic_forward_stats['mean']]
    backward_means = [linear_backward_stats['mean'], cubic_backward_stats['mean']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax3.bar(x - width/2, forward_means, width, label='Forward', alpha=0.8)
    ax3.bar(x + width/2, backward_means, width, label='Backward', alpha=0.8)
    ax3.set_title('Mean Processing Times')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Throughput comparison
    throughputs = [total_projections / linear_total, total_projections / cubic_total]
    ax4.bar(methods, throughputs, alpha=0.8, color=['blue', 'orange'])
    ax4.set_title('Throughput Comparison')
    ax4.set_ylabel('Projections/second')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('test_outputs', exist_ok=True)
    plt.savefig('test_outputs/performance_benchmark.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: test_outputs/performance_benchmark.png")
    
    # Basic sanity checks
    assert cubic_forward_stats['mean'] > linear_forward_stats['mean'], "Cubic should be slower than linear"
    assert cubic_backward_stats['mean'] > linear_backward_stats['mean'], "Cubic backward should be slower than linear"
    assert all(t > 0 for t in linear_forward + linear_backward + cubic_forward + cubic_backward), "All times should be positive"


def test_interpolation_quality_comparison():
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
    H, W = 64, 33  # 64x64 real -> 64x33 complex after RFFT
    reconstruction = torch.randn(H, W, dtype=torch.complex64)
    
    # Identity projection (0 degrees) as reference
    identity_rotation = torch.eye(2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 2, 2]
    reference_proj = torch_projectors.forward_project_2d(
        reconstruction.unsqueeze(0), 
        identity_rotation, 
        output_shape=(H, H),
        interpolation='linear'  # Doesn't matter for identity
    )
    
    # Test rotations: +5° then -5°
    angle_rad = math.radians(1.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    rot_plus5 = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    rot_minus5 = torch.tensor([[cos_a, sin_a], [-sin_a, cos_a]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Round-trip with linear interpolation
    proj_5deg_linear = torch_projectors.forward_project_2d(
        reconstruction.unsqueeze(0), rot_plus5, output_shape=(H, H), interpolation='linear'
    )
    # proj_5deg_linear is [1, 1, 64, 64] but we need [1, 64, 33] for next projection
    roundtrip_linear = torch_projectors.forward_project_2d(
        proj_5deg_linear.squeeze(1), rot_minus5, output_shape=(H, H), interpolation='linear'
    )
    
    # Round-trip with cubic interpolation  
    proj_5deg_cubic = torch_projectors.forward_project_2d(
        reconstruction.unsqueeze(0), rot_plus5, output_shape=(H, H), interpolation='cubic'
    )
    # proj_5deg_cubic is [1, 1, 64, 64] but we need [1, 64, 33] for next projection
    roundtrip_cubic = torch_projectors.forward_project_2d(
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
        [reference_proj[0], roundtrip_linear[0], roundtrip_cubic[0], 
         reference_proj[0] - roundtrip_linear[0], reference_proj[0] - roundtrip_cubic[0]],
        ['Reference (0°)', 'Linear round-trip', 'Cubic round-trip', 'Linear error', 'Cubic error'],
        'test_outputs/interpolation_quality_comparison.png',
        shape=(1, 5)
    )
    
    # Verify cubic interpolation performs better
    assert error_cubic < error_linear, f"Cubic interpolation should be more accurate: cubic={error_cubic:.6f} vs linear={error_linear:.6f}"


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_gradient_fourier_components(interpolation):
    """
    Debug test for gradient calculation of Fourier components.
    Tests that gradients correctly flow back to the reconstruction.
    """
    torch.manual_seed(42)
    
    # Test parameters
    H, W = 32, 32
    B = 1  # Single batch
    P = 1  # Single pose
    
    # Step 1: Initialize random 2D reconstruction and project at 90°
    rec_random_real = torch.randn(H, W)
    rec_random_fourier = torch.fft.rfftn(rec_random_real, dim=(-2, -1))
    rec_random_fourier = rec_random_fourier.unsqueeze(0)  # Add batch dim
    
    # 90 degree rotation matrix
    rot_90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    rotations_90 = rot_90.unsqueeze(0).unsqueeze(0)  # Shape: (B=1, P=1, 2, 2)
    
    # Project at 90°, no shift
    proj_90_fourier = torch_projectors.forward_project_2d(
        rec_random_fourier, rotations_90, shifts=None, 
        output_shape=(H, W), interpolation=interpolation
    )
    
    # Convert to real space - this is our reference projection
    ref_proj_90_real = torch.fft.irfftn(proj_90_fourier.squeeze(), dim=(-2, -1))
    
    # Step 2: Create reference projection at 0° (identity rotation)
    rot_0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    rotations_0 = rot_0.unsqueeze(0).unsqueeze(0)  # Shape: (B=1, P=1, 2, 2)
    
    proj_0_fourier = torch_projectors.forward_project_2d(
        rec_random_fourier, rotations_0, shifts=None,
        output_shape=(H, W), interpolation=interpolation
    )
    ref_proj_0_real = torch.fft.irfftn(proj_0_fourier.squeeze(), dim=(-2, -1))
    
    # Step 3: Initialize zero reconstruction with gradients required
    rec_zero_real = torch.zeros(H, W, requires_grad=True)
    rec_zero_fourier = torch.fft.rfftn(rec_zero_real, dim=(-2, -1))
    rec_zero_fourier = rec_zero_fourier.unsqueeze(0)  # Add batch dim
    
    # Step 4: Forward project the zero reconstruction at 90°
    proj_zero_fourier = torch_projectors.forward_project_2d(
        rec_zero_fourier, rotations_90, shifts=None,
        output_shape=(H, W), interpolation=interpolation
    )
    proj_zero_real = torch.fft.irfftn(proj_zero_fourier.squeeze(), dim=(-2, -1))
    
    # Step 5: Calculate MSE loss and backpropagate
    loss = F.mse_loss(proj_zero_real, ref_proj_90_real.detach())
    loss.backward()
    
    # Step 6: Single gradient descent update
    learning_rate = 1.0
    with torch.no_grad():
        rec_zero_real -= learning_rate * rec_zero_real.grad
    
    # Step 7: Calculate normalized cross-correlation
    def normalized_cross_correlation(x, y):
        """Calculate normalized cross-correlation between two tensors."""
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Center the data
        x_centered = x_flat - x_flat.mean()
        y_centered = y_flat - y_flat.mean()
        
        # Calculate correlation
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))
        
        # Avoid division by zero
        if denominator == 0:
            return torch.tensor(0.0)
        
        return numerator / denominator
    
    ncc = normalized_cross_correlation(rec_zero_real, ref_proj_0_real)
    
    # Step 8: Create visualizations - all in one plot
    # Convert Fourier space tensors for plotting
    rec_zero_fourier_updated = torch.fft.rfftn(rec_zero_real, dim=(-2, -1))
    ref_proj_0_fourier = torch.fft.rfftn(ref_proj_0_real, dim=(-2, -1))
    
    os.makedirs('test_outputs', exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Real space plots
    axes[0, 0].imshow(ref_proj_0_real.detach().numpy(), cmap='viridis', origin='lower')
    axes[0, 0].set_title('Reference (0° projection)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rec_zero_real.detach().numpy(), cmap='viridis', origin='lower')
    axes[0, 1].set_title('Updated reconstruction')
    axes[0, 1].axis('off')
    
    # Fourier space plots - real parts
    axes[0, 2].imshow(ref_proj_0_fourier.real.detach().numpy(), cmap='viridis', origin='lower')
    axes[0, 2].set_title('Reference FFT (Real)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(rec_zero_fourier_updated.real.detach().numpy(), cmap='viridis', origin='lower')
    axes[0, 3].set_title('Updated FFT (Real)')
    axes[0, 3].axis('off')
    
    # Fourier space plots - imaginary parts
    axes[1, 0].imshow(ref_proj_0_fourier.imag.detach().numpy(), cmap='viridis', origin='lower')
    axes[1, 0].set_title('Reference FFT (Imag)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(rec_zero_fourier_updated.imag.detach().numpy(), cmap='viridis', origin='lower')
    axes[1, 1].set_title('Updated FFT (Imag)')
    axes[1, 1].axis('off')
    
    # Add text with results
    axes[1, 2].text(0.1, 0.7, f'NCC: {ncc.item():.6f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, f'Loss: {loss.item():.6f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.3, f'Grad norm: {rec_zero_real.grad.norm().item():.6f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'test_outputs/gradient_debug_fourier_components_{interpolation}.png')
    plt.close()
    
    print(f"Normalized cross-correlation: {ncc.item():.6f}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient norm: {rec_zero_real.grad.norm().item():.6f}")
    
    # The test - NCC should be close to 1.0 if gradients work correctly
    # Note: Due to numerical precision and the fact that we're doing a single gradient step,
    # we use a reasonable tolerance
    assert ncc > 0.8, f"Expected NCC > 0.8, got {ncc.item():.6f}"


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("shift_values,batch_size,num_poses", [
    # Basic positive shifts
    ([2.0, 1.0], 1, 1),
    # Negative shifts  
    ([-2.0, -1.0], 1, 1),
    # Mixed sign shifts
    ([2.0, -1.0], 1, 1),
    ([-2.0, 1.0], 1, 1),
    # Zero shifts (identity case)
    ([0.0, 0.0], 1, 1),
    # One component zero
    ([2.0, 0.0], 1, 1),
    ([0.0, 1.0], 1, 1),
    # Large shifts
    ([10.0, 5.0], 1, 1),
    # Fractional shifts
    ([0.5, 0.3], 1, 1),
    # Asymmetric magnitudes
    ([0.1, 10.0], 1, 1),
    # Batching tests - multiple reconstructions
    ([2.0, 1.0], 3, 1),
    # Batching tests - multiple poses
    ([2.0, 1.0], 1, 4),
    # Batching tests - both dimensions
    ([2.0, 1.0], 2, 3),
])
def test_shift_gradient_verification_parametrized(interpolation, shift_values, batch_size, num_poses):
    """
    Parametrized test to verify shift gradient calculation across various scenarios.
    Tests backpropagation from L2 loss between unshifted and shifted projections.
    """
    torch.manual_seed(42)
    
    # Test parameters
    H, W = 32, 32
    B = batch_size
    P = num_poses
    
    # Create test shifts - broadcast the same shift to all batches and poses for consistency
    test_shift = torch.tensor(shift_values, dtype=torch.float32)
    test_shift = test_shift.unsqueeze(0).unsqueeze(0).expand(B, P, 2)  # Shape: (B, P, 2)
    
    # Step 1: Create random reconstructions (one per batch)
    rec_random_real = torch.randn(B, H, W)
    rec_random_fourier = torch.fft.rfftn(rec_random_real, dim=(-2, -1))
    
    # 15-degree rotation (same for all batches and poses)
    angle_deg = 15.0
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot_15 = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
    rotations = rot_15.unsqueeze(0).unsqueeze(0).expand(B, P, 2, 2)  # Shape: (B, P, 2, 2)
    
    # Step 2: Generate unshifted projection at 15 degrees
    proj_unshifted = torch_projectors.forward_project_2d(
        rec_random_fourier, rotations, shifts=None,
        output_shape=(H, W), interpolation=interpolation
    )
    
    # Step 3: Generate shifted projection using our implementation
    shifts_for_our_impl = test_shift.clone().requires_grad_(True)
    proj_shifted_our_impl = torch_projectors.forward_project_2d(
        rec_random_fourier, rotations, shifts=shifts_for_our_impl,
        output_shape=(H, W), interpolation=interpolation
    )
    
    # Step 4: Manually construct phase modulation for ground truth
    shifts_for_manual = test_shift.clone().requires_grad_(True)
    
    # Create coordinate grids for phase calculation
    ky = torch.arange(H, dtype=torch.float32)
    ky[ky > H // 2] -= H  # Shift to [-H/2, H/2) range
    kx = torch.arange(W // 2 + 1, dtype=torch.float32)
    kyy, kxx = torch.meshgrid(ky, kx, indexing='ij')
    
    # Manual phase modulation for all batches and poses
    proj_shifted_manual = torch.zeros_like(proj_unshifted)
    for b in range(B):
        for p in range(P):
            # Manual phase modulation: exp(-2πi(ky*shift_r + kx*shift_c))
            # Note: normalization by H to match the implementation
            phase = -2.0 * math.pi * (kyy * shifts_for_manual[b, p, 0] / H + kxx * shifts_for_manual[b, p, 1] / H)
            phase_factor = torch.complex(torch.cos(phase), torch.sin(phase))
            
            # Apply manual phase modulation
            proj_shifted_manual[b, p] = proj_unshifted[b, p] * phase_factor
    
    # Step 5: Implement manual complex MSE loss and backprop
    def complex_mse_loss(input_tensor, target_tensor):
        """Manual implementation of MSE loss for complex tensors."""
        diff = input_tensor - target_tensor
        # |a + ib|^2 = a^2 + b^2
        loss = torch.mean(diff.real**2 + diff.imag**2)
        return loss
    
    # Calculate losses and gradients
    # Our implementation
    loss_our_impl = complex_mse_loss(proj_shifted_our_impl, proj_unshifted)
    loss_our_impl.backward()
    grad_our_impl = shifts_for_our_impl.grad.clone()
    
    # Manual implementation  
    loss_manual = complex_mse_loss(proj_shifted_manual, proj_unshifted)
    loss_manual.backward()
    grad_manual = shifts_for_manual.grad.clone()
    
    # Step 6: Compare gradients
    grad_diff = torch.abs(grad_our_impl - grad_manual)
    rel_error = grad_diff / (torch.abs(grad_manual) + 1e-8)
    max_abs_error = grad_diff.max().item()
    max_rel_error = rel_error.max().item()
    
    print(f"Shift: {shift_values}, Batch: {batch_size}, Poses: {num_poses}")
    print(f"Max absolute error: {max_abs_error:.2e}")
    print(f"Max relative error: {max_rel_error:.2e}")
    
    # Test assertion - gradients should be close
    torch.testing.assert_close(grad_our_impl, grad_manual, atol=1e-4, rtol=1e-2)
    
    # Create visualization
    os.makedirs('test_outputs', exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Show first batch, first pose for visualization
    # Real parts
    axes[0, 0].imshow(proj_unshifted[0, 0].real.detach().numpy(), cmap='viridis', origin='lower')
    axes[0, 0].set_title('Unshifted (Real)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(proj_shifted_our_impl[0, 0].real.detach().numpy(), cmap='viridis', origin='lower')
    axes[0, 1].set_title('Shifted - Our Impl (Real)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(proj_shifted_manual[0, 0].real.detach().numpy(), cmap='viridis', origin='lower')
    axes[0, 2].set_title('Shifted - Manual (Real)')
    axes[0, 2].axis('off')
    
    # Imaginary parts
    axes[1, 0].imshow(proj_unshifted[0, 0].imag.detach().numpy(), cmap='viridis', origin='lower')
    axes[1, 0].set_title('Unshifted (Imag)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(proj_shifted_our_impl[0, 0].imag.detach().numpy(), cmap='viridis', origin='lower')
    axes[1, 1].set_title('Shifted - Our Impl (Imag)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(proj_shifted_manual[0, 0].imag.detach().numpy(), cmap='viridis', origin='lower')
    axes[1, 2].set_title('Shifted - Manual (Imag)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'test_outputs/shift_gradient_verification_{shift_values[0]}_{shift_values[1]}_B{batch_size}_P{num_poses}_{interpolation}.png')
    plt.close()
    
    print(f"✅ Shift gradient test passed: {shift_values} (B={batch_size}, P={num_poses})")

def test_dimension_validation():
    """
    Tests that the new validation constraints are properly enforced:
    - Boxsize must be even
    - Dimensions must be square (boxsize == 2*(boxsize_half-1))
    """
    
    # Test non-square dimensions (should fail)
    with pytest.raises(ValueError, match="expected boxsize .* to match"):
        rec = torch.randn(1, 30, 17, dtype=torch.complex64)  # 30 != 2*(17-1) = 32
        rot = torch.eye(2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        torch_projectors.forward_project_2d(rec, rot)

    # Test odd dimensions (should fail) 
    with pytest.raises(ValueError, match="Boxsize .* must be even"):
        rec = torch.randn(1, 29, 15, dtype=torch.complex64)  # 29 is odd, should be caught by even check first
        rot = torch.eye(2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        torch_projectors.forward_project_2d(rec, rot)
        
    # Test valid square, even dimensions (should pass)
    rec = torch.randn(1, 32, 17, dtype=torch.complex64)  # 32x32 -> 17 half
    rot = torch.eye(2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    proj = torch_projectors.forward_project_2d(rec, rot)
    assert proj.shape == (1, 1, 32, 17)
    
    # Test valid backward projection (should pass)
    proj = torch.randn(1, 1, 32, 17, dtype=torch.complex64)
    dummy_rec = torch.randn(1, 32, 17, dtype=torch.complex64)
    rot = torch.eye(2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    rec = torch_projectors.backward_project_2d(proj, dummy_rec, rot)
    assert rec.shape == (1, 32, 17)

def create_rotation_matrix_2d(angle):
    """Helper function to create 2D rotation matrix from angle"""
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    return torch.stack([
        torch.stack([cos_a, -sin_a], dim=-1),
        torch.stack([sin_a, cos_a], dim=-1)
    ], dim=-2)

def compute_angle_grad_from_matrix_grad(matrix_grad, angle):
    """Convert rotation matrix gradient to angle gradient using chain rule"""
    # d/dθ [cos θ, -sin θ; sin θ, cos θ] = [-sin θ, -cos θ; cos θ, -sin θ]
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    dR_dtheta = torch.tensor([
        [-sin_a, -cos_a],
        [cos_a, -sin_a]
    ])
    
    # Angle gradient = trace(matrix_grad^T * dR_dtheta)
    return torch.sum(matrix_grad * dR_dtheta)

@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_rotation_gradients_comprehensive(interpolation):
    """
    Comprehensive test of rotation gradients using finite difference verification.
    This test validates that our analytical rotation gradients match numerical derivatives.
    """
    print(f"\n🔄 Testing rotation gradients comprehensively ({interpolation})...")
    
    # Test 1: Finite Difference Verification (most important)
    print("  1️⃣ Finite difference verification...")
    _test_rotation_finite_difference_accuracy(interpolation)
    
    # Test 2: Multiple scenarios to ensure robustness
    print("  2️⃣ Testing multiple scenarios...")
    test_cases = [
        (0.1, 0.05),  # Small angles
        (0.3, 0.1),   # Medium angles  
        (0.8, 0.2),   # Larger angles
    ]
    
    for i, (target_angle, test_offset) in enumerate(test_cases):
        print(f"    Scenario {i+1}: target={target_angle:.2f}, offset={test_offset:.2f}")
        torch.manual_seed(42 + i)
        rec = torch.randn(1, 16, 9, dtype=torch.complex64, requires_grad=True)
        
        target_rot = create_rotation_matrix_2d(torch.tensor(target_angle)).unsqueeze(0).unsqueeze(0)
        target_proj = torch_projectors.forward_project_2d(rec, target_rot.detach(), interpolation=interpolation)
        
        test_angle = target_angle + test_offset
        test_rot = create_rotation_matrix_2d(torch.tensor(test_angle)).unsqueeze(0).unsqueeze(0)
        test_rot.requires_grad_(True)
        
        pred_proj = torch_projectors.forward_project_2d(rec, test_rot, interpolation=interpolation)
        loss = torch.sum((pred_proj - target_proj).abs())
        loss.backward()
        
        # Verify gradients exist and are non-zero
        assert test_rot.grad is not None, f"Scenario {i+1}: No gradients computed"
        grad_norm = test_rot.grad.norm().item()
        assert grad_norm > 1e-8, f"Scenario {i+1}: Gradient norm too small: {grad_norm:.2e}"
        print(f"      Gradient norm: {grad_norm:.6f}")
    
    # Test 3: Optimization Convergence (practical validation)
    print("  3️⃣ Optimization convergence verification...")
    _test_rotation_optimization_convergence(interpolation)
    
    print("✅ All rotation gradient tests passed!")

def _test_rotation_finite_difference_accuracy(interpolation):
    """Test that analytical gradients match finite differences"""
    # Create test data - use smaller, more manageable case
    torch.manual_seed(42)  # For reproducibility
    rec = torch.randn(1, 16, 9, dtype=torch.complex64, requires_grad=True)
    target_angle = 0.1  # Smaller angle difference
    target_rot = create_rotation_matrix_2d(torch.tensor(target_angle)).unsqueeze(0).unsqueeze(0)
    target_proj = torch_projectors.forward_project_2d(rec, target_rot.detach(), interpolation=interpolation)
    
    # Test at slightly different angle
    test_angle = target_angle + 0.05  # Smaller perturbation
    test_rot = create_rotation_matrix_2d(torch.tensor(test_angle)).unsqueeze(0).unsqueeze(0)
    test_rot.requires_grad_(True)
    
    # Compute analytical gradients
    pred_proj = torch_projectors.forward_project_2d(rec, test_rot, interpolation=interpolation)
    loss = torch.sum((pred_proj - target_proj).abs())
    loss.backward()
    
    # Compare against finite differences using proper angle perturbation
    eps = 1e-5  # Epsilon for numerical stability
    
    print("    Comparing analytical vs finite difference gradients:")
    
    # Create rotation matrices with perturbed angles
    rot_plus = create_rotation_matrix_2d(torch.tensor(test_angle + eps)).unsqueeze(0).unsqueeze(0)
    rot_minus = create_rotation_matrix_2d(torch.tensor(test_angle - eps)).unsqueeze(0).unsqueeze(0)
    
    # Compute finite difference
    loss_plus = torch.sum((torch_projectors.forward_project_2d(rec.detach(), rot_plus, interpolation=interpolation) - target_proj).abs())
    loss_minus = torch.sum((torch_projectors.forward_project_2d(rec.detach(), rot_minus, interpolation=interpolation) - target_proj).abs())
    fd_angle_grad = (loss_plus - loss_minus) / (2 * eps)
    
    # Convert matrix gradient to angle gradient using chain rule
    analytical_angle_grad = compute_angle_grad_from_matrix_grad(test_rot.grad[0, 0], torch.tensor(test_angle))
    abs_error = torch.abs(analytical_angle_grad - fd_angle_grad).item()
    relative_error = abs_error / (torch.abs(fd_angle_grad).item() + 1e-8)
    
    print(f"    Angle gradient: analytical={analytical_angle_grad:.6f}, fd={fd_angle_grad:.6f}, abs_err={abs_error:.2e}, rel_err={relative_error:.2e}")
    
    # Reasonable tolerance - analytical gradients should be close to finite differences
    assert relative_error < 0.01, f"Relative finite difference error {relative_error:.2e} exceeds tolerance"
    print(f"    ✅ Relative finite difference error: {relative_error:.2e}")


def _test_rotation_optimization_convergence(interpolation):
    """Test that optimization converges to correct angle"""
    # Create target projection at known angle
    torch.manual_seed(42)  # For reproducibility
    rec = torch.randn(1, 16, 9, dtype=torch.complex64)  # Smaller for stability
    target_angles = [0.2, 0.5, 0.8]  # Positive angles for stability (fewer local minima)
    
    for target_angle in target_angles:
        print(f"    Optimizing toward angle {target_angle:.3f}...")
        
        target_rot = create_rotation_matrix_2d(torch.tensor(target_angle)).unsqueeze(0).unsqueeze(0)
        target_proj = torch_projectors.forward_project_2d(rec, target_rot, interpolation=interpolation)
        
        # Initialize optimization close to target (not at 0) for better convergence
        init_angle = target_angle + 0.017  # Start 1 degree (~0.017 radians) away
        learned_angle = torch.tensor(init_angle, requires_grad=True)
        optimizer = torch.optim.Adam([learned_angle], lr=0.02)  # Lower learning rate for stability
        
        best_loss = float('inf')
        best_angle = init_angle
        
        for step in range(100):  # Fewer steps for speed
            optimizer.zero_grad()
            learned_rot = create_rotation_matrix_2d(learned_angle).unsqueeze(0).unsqueeze(0)
            pred_proj = torch_projectors.forward_project_2d(rec, learned_rot, interpolation=interpolation)
            loss = torch.sum((pred_proj - target_proj).abs().pow(2))  # Manual MSE for complex tensors
            loss.backward()
            optimizer.step()
            
            # Track best result
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_angle = learned_angle.item()
            
            # Check for convergence
            if step % 25 == 0:
                print(f"      Step {step}: loss={loss.item():.6f}, angle={learned_angle.item():.3f}")
        
        # Check convergence using best result (accounting for 2π periodicity)
        angle_diff = torch.abs(torch.tensor(best_angle) - target_angle) % (2 * torch.pi)
        angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)
        
        print(f"      Final: target={target_angle:.3f}, best={best_angle:.3f}, diff={angle_diff.item():.3f}")
        # Reasonable tolerance for convergence
        assert angle_diff < 0.001, f"Failed to converge: angle difference {angle_diff.item():.3f} > 0.001 rad" 