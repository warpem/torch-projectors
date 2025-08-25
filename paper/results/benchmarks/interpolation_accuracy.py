#!/usr/bin/env python3
"""
Benchmark script for interpolation accuracy using real shell correlation.

This script measures how well interpolation preserves image quality across 
different spatial frequencies by comparing projections generated with 
different oversampling factors. Uses a fine checkerboard pattern to 
provide high-frequency content, and measures real shell correlation 
between low and high oversampling projections.

Usage:
    python interpolation_accuracy.py --platform-name "m2-mps" --device mps
    python interpolation_accuracy.py --platform-name "a100-cuda" --device auto
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch_projectors
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def parse_device(device_str: str) -> torch.device:
    """Parse device string and return appropriate torch.device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_str == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    elif device_str == 'mps':
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    elif device_str == 'cpu':
        return torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: {device_str}")


def create_checkerboard_pattern(size: int, device: torch.device) -> torch.Tensor:
    """Create a fine checkerboard pattern where each cell is 1 pixel.
    
    Args:
        size: Size of the square image (height and width)
        device: PyTorch device
        
    Returns:
        2D real tensor of shape [size, size] with values 0 and 1
    """
    # Create coordinate grids
    y = torch.arange(size, device=device, dtype=torch.float32)
    x = torch.arange(size, device=device, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Create checkerboard: alternating pattern based on sum of coordinates
    checkerboard = torch.rand_like(Y, device='cpu').to(device) - 0.5
    
    return checkerboard


def apply_lowpass_filter(image: torch.Tensor, cutoff_freq: float = 0.5) -> torch.Tensor:
    """Apply low-pass filter to band-limit the image to Nyquist frequency.
    
    Args:
        image: 2D real image tensor [height, width]
        cutoff_freq: Cutoff frequency as fraction of Nyquist (0.5 = Nyquist frequency)
        
    Returns:
        2D real tensor with filtered content
    """
    # Transform to Fourier space
    fourier = torch.fft.fftshift(torch.fft.fft2(image))
    
    # Create frequency coordinate grid
    height, width = image.shape
    center_y, center_x = height // 2, width // 2
    
    y = torch.arange(height, device=image.device, dtype=torch.float32) - center_y
    x = torch.arange(width, device=image.device, dtype=torch.float32) - center_x
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Calculate normalized frequency radius
    freq_radius = torch.sqrt(X**2 + Y**2) / min(height, width)
    
    # Create low-pass filter (smooth cutoff with 2-pixel transition)
    filter_mask = torch.clamp((cutoff_freq - freq_radius) / (2.0 / min(height, width)), 0.0, 1.0)
    
    # Apply filter
    filtered_fourier = fourier * filter_mask
    
    # Transform back to real space
    filtered_image = torch.fft.ifft2(torch.fft.ifftshift(filtered_fourier)).real
    
    return filtered_image


def create_radial_mask(size: int, diameter: float, soft_edge: float, device: torch.device) -> torch.Tensor:
    """Create a circular mask with soft edge falloff.
    
    Args:
        size: Size of the square image
        diameter: Diameter of the circular mask
        soft_edge: Width of the soft edge transition in pixels
        device: PyTorch device
        
    Returns:
        2D real tensor of shape [size, size] with values between 0 and 1
    """
    center = size / 2.0
    radius = diameter / 2.0
    
    # Create coordinate grids centered at the image center
    y = torch.arange(size, device=device, dtype=torch.float32) - center + 0.5
    x = torch.arange(size, device=device, dtype=torch.float32) - center + 0.5
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Calculate distance from center
    r = torch.sqrt(X**2 + Y**2)
    
    # Create mask with soft edge falloff
    # Inside radius: 1, outside radius+soft_edge: 0, linear transition in between
    mask = torch.clamp((radius + soft_edge - r) / soft_edge, 0.0, 1.0)
    
    return mask


def prepare_image_for_projection(image: torch.Tensor, target_size: int, device: torch.device) -> torch.Tensor:
    """Prepare image for projection: pad, fftshift, and transform to Fourier space.
    
    Args:
        image: 2D real image tensor
        target_size: Target size for padding (should be larger than original)
        device: PyTorch device
        
    Returns:
        2D complex tensor in RFFT format [target_size, target_size//2 + 1]
    """
    original_size = image.shape[0]
    
    if target_size <= original_size:
        padded = image
    else:
        # Zero-pad to target size
        pad_size = (target_size - original_size) // 2
        padded = torch.nn.functional.pad(image, (pad_size, pad_size, pad_size, pad_size), value=0.0)
    
    # Apply fftshift before FFT for proper rotation centering
    shifted = torch.fft.fftshift(padded)
    
    # Transform to Fourier space using rfft2
    fourier = torch.fft.rfft2(shifted)
    
    return fourier


def generate_random_rotations(num_angles: int, device: torch.device) -> torch.Tensor:
    """Generate random 2D rotation matrices.
    
    Args:
        num_angles: Number of random rotation angles to generate
        device: PyTorch device
        
    Returns:
        Tensor of shape [1, num_angles, 2, 2] containing rotation matrices
    """
    # Generate random angles between 0 and 2π
    angles = torch.rand(num_angles, device='cpu').to(device) * 2 * math.pi
    #angles[0] = 0.0
    
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    
    # Create rotation matrices
    rotations = torch.zeros(1, num_angles, 2, 2, device=device)
    rotations[0, :, 0, 0] = cos_a
    rotations[0, :, 0, 1] = -sin_a
    rotations[0, :, 1, 0] = sin_a
    rotations[0, :, 1, 1] = cos_a
    
    return rotations


def calculate_fourier_shell_correlation(fourier_proj1: torch.Tensor, fourier_proj2: torch.Tensor,
                                      max_radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Fourier shell correlation between two sets of Fourier projections.
    
    Args:
        fourier_proj1: First set of Fourier projections [num_proj, height, width/2+1]  
        fourier_proj2: Second set of Fourier projections [num_proj, height, width/2+1]
        max_radius: Maximum radius for shell correlation calculation
        
    Returns:
        Tuple of (radii, correlations) where:
        - radii: Array of shell radii in frequency units
        - correlations: Array of FSC values for each shell
    """
    _, height, width_half = fourier_proj1.shape
    width = (width_half - 1) * 2  # Reconstruct full width
    center = height / 2.0
    
    # Create proper frequency coordinate grids using PyTorch frequency functions
    x_freqs = torch.fft.rfftfreq(width, device=fourier_proj1.device) * width  # Scale by width to get pixel frequencies
    y_freqs = torch.fft.fftfreq(height, device=fourier_proj1.device) * height  # Scale by height to get pixel frequencies
    Y, X = torch.meshgrid(y_freqs, x_freqs, indexing='ij')
    
    # Calculate frequency radius in pixel units and round to nearest integer
    freq_radius = torch.sqrt(X**2 + Y**2)
    freq_radius_int = torch.round(freq_radius).int()
    
    radii = []
    correlations = []
    
    # Calculate FSC for each integer shell radius
    for shell_radius in range(0, max_radius + 1):
        # Create shell mask for this integer radius
        shell_mask = (freq_radius_int == shell_radius)
        
        if not shell_mask.any():
            continue
            
        # Extract complex values from all projections for this shell
        values1 = fourier_proj1[:, shell_mask].flatten()  # [num_proj * num_shell_pixels]
        values2 = fourier_proj2[:, shell_mask].flatten()  # [num_proj * num_shell_pixels]
        
        if len(values1) == 0:
            continue
            
        # Calculate FSC: normalized cross-correlation of complex values
        # FSC = |sum(F1 * conj(F2))| / sqrt(sum(|F1|^2) * sum(|F2|^2))
        cross_correlation = (values1 * torch.conj(values2)).sum()
        power1 = (values1.abs()**2).sum()
        power2 = (values2.abs()**2).sum()
        
        if power1 > 0 and power2 > 0:
            fsc = (cross_correlation.abs() / torch.sqrt(power1 * power2)).item()
        else:
            fsc = 0.0

        radii.append(float(shell_radius))
        correlations.append(fsc)
    
    return np.array(radii), np.array(correlations)


def calculate_bilinear_filter_response(radii: np.ndarray, image_size: int) -> np.ndarray:
    """Calculate theoretical bilinear filter response curve.
    
    Args:
        radii: Array of shell radii in pixels
        image_size: Image size to convert radii to normalized frequencies
        
    Returns:
        Array of theoretical bilinear filter response values
    """
    # Convert pixel radii to normalized frequencies (0 to 0.5)
    fr_values = radii / image_size
    
    # Generate the theoretical bilinear filter response curve
    def integrand(phi, fr):
        return (1 - (1/3) * (1 - np.cos(2 * np.pi * fr * np.cos(phi)))) * \
               (1 - (1/3) * (1 - np.cos(2 * np.pi * fr * np.sin(phi))))
    
    combined_curve = []
    for fr in fr_values:
        if fr == 0:
            # At DC, the response should be 1.0
            combined_curve.append(1.0)
        else:
            # Calculate integral for this frequency
            result, _ = quad(integrand, 0, 2 * np.pi, args=(fr,))
            combined_curve.append(np.sqrt(result / (2 * np.pi)))
    
    return np.array(combined_curve)


def calculate_real_shell_correlation(projections1: torch.Tensor, projections2: torch.Tensor, 
                                   max_radius: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate real shell correlation and relative intensity between two sets of projections.
    
    Args:
        projections1: First set of projections [num_proj, height, width] (low oversampling)
        projections2: Second set of projections [num_proj, height, width] (high oversampling)
        max_radius: Maximum radius for shell correlation calculation
        
    Returns:
        Tuple of (radii, correlations, intensity_ratios) where:
        - radii: Array of shell radii
        - correlations: Array of correlation values for each shell
        - intensity_ratios: Array of RMS intensity ratios (projections1 / projections2) for each shell
    """
    _, height, width = projections1.shape
    center = height / 2.0
    
    # Create coordinate grids for radius calculation
    y = torch.arange(height, device=projections1.device, dtype=torch.float32) - center + 0.5
    x = torch.arange(width, device=projections1.device, dtype=torch.float32) - center + 0.5
    Y, X = torch.meshgrid(y, x, indexing='ij')
    radius_map = torch.sqrt(X**2 + Y**2)
    
    # Calculate correlations and intensity ratios for each shell radius
    radii = []
    correlations = []
    intensity_ratios = []
    
    for r in range(1, max_radius + 1):
        # Create mask for current shell (radius between r-0.5 and r+0.5)
        shell_mask = (radius_map >= r - 0.5) & (radius_map < r + 0.5)
        
        if not shell_mask.any():
            continue
            
        # Extract values from all projections for this shell
        values1 = projections1[:, shell_mask].flatten()  # [num_proj * num_shell_pixels]
        values2 = projections2[:, shell_mask].flatten()  # [num_proj * num_shell_pixels]
        
        if len(values1) == 0:
            continue
            
        # Calculate normalized cross-correlation
        mean1 = values1.mean()
        mean2 = values2.mean()
        
        centered1 = values1 - mean1
        centered2 = values2 - mean2
        
        numerator = (centered1 * centered2).sum()
        denominator = torch.sqrt((centered1**2).sum() * (centered2**2).sum())
        
        if denominator > 0:
            correlation = (numerator / denominator).item()
        else:
            correlation = 0.0
            
        # Calculate RMS intensity ratio (low oversampling / high oversampling)
        rms1 = torch.sqrt((values1**2).mean())
        rms2 = torch.sqrt((values2**2).mean())
        
        if rms2 > 0:
            intensity_ratio = (rms1 / rms2).item()
        else:
            intensity_ratio = 0.0
            
        radii.append(r)
        correlations.append(correlation)
        intensity_ratios.append(intensity_ratio)
    
    return np.array(radii), np.array(correlations), np.array(intensity_ratios)


def run_interpolation_benchmark(device: torch.device) -> Dict[str, Any]:
    """Run the interpolation accuracy benchmark.
    
    Args:
        device: PyTorch device
        
    Returns:
        Dictionary containing benchmark results
    """
    print("Generating test pattern...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    base_size = 64
    reference_oversampling = 30
    padded_size = base_size * reference_oversampling
    mask_diameter = 62
    soft_edge = 1
    num_angles = 128
    interpolation_methods = ['linear', 'cubic']
    test_oversampling_levels = [1.0, 1.5, 2.0]
    
    # Step 1: Create checkerboard pattern
    checkerboard = create_checkerboard_pattern(base_size, device)
    
    # Step 2: Apply low-pass filter to band-limit to Nyquist frequency
    print("Applying low-pass filter...")
    filtered_checkerboard = apply_lowpass_filter(checkerboard, cutoff_freq=0.5)
    
    # Step 3: Apply radial mask
    mask = create_radial_mask(base_size, mask_diameter, soft_edge, device)
    masked_image = filtered_checkerboard * mask
    
    print("Preparing images for projection...")
    
    # Prepare images for different oversampling levels
    image_fourier_1x = prepare_image_for_projection(masked_image, base_size, device)  # 64x64 for 1x oversampling
    image_fourier_1_5x = prepare_image_for_projection(masked_image, int(base_size * 1.5), device)  # 96x96 for 1.5x oversampling
    image_fourier_2x = prepare_image_for_projection(masked_image, base_size * 2, device)  # 128x128 for 2x oversampling
    image_fourier_reference = prepare_image_for_projection(masked_image, padded_size, device)  # 1920x1920 for 30x reference
    
    print("Generating rotation angles...")
    
    # Step 4: Generate random rotation angles
    rotations = generate_random_rotations(num_angles, device)
    
    # Generate reference projections with 30x oversampling and linear interpolation
    print("Generating reference projections (30x oversampling, linear interpolation)...")
    reference_projections = torch_projectors.project_2d_forw(
        image_fourier_reference.unsqueeze(0),  # Add batch dimension
        rotations,
        shifts=None,
        output_shape=(base_size, base_size),
        interpolation='linear',
        oversampling=reference_oversampling
    ).squeeze(0)  # Remove batch dimension
    
    results = {}
    debug_images = {
        "original_checkerboard": checkerboard.cpu().numpy().tolist(),
        "filtered_checkerboard": filtered_checkerboard.cpu().numpy().tolist(),
        "masked_image": masked_image.cpu().numpy().tolist()
    }
    
    # Convert reference to real space for debug images
    reference_real = torch.fft.irfft2(reference_projections, s=(base_size, base_size))
    reference_real = torch.fft.ifftshift(reference_real, dim=(-2, -1))
    debug_images["first_rotation_reference_30x"] = reference_real[0].cpu().numpy().tolist()
    
    # Test each interpolation method against the reference
    for interp_method in interpolation_methods:
        print(f"\nTesting {interp_method} interpolation...")
        
        method_results = {}
        
        # Test each oversampling level against the reference
        for oversampling in test_oversampling_levels:
            print(f"  Projecting with {oversampling}x oversampling...")
            
            # Select the appropriate image size for this oversampling level
            if oversampling == 1.0:
                image_to_use = image_fourier_1x
            elif oversampling == 1.5:
                image_to_use = image_fourier_1_5x
            elif oversampling == 2.0:
                image_to_use = image_fourier_2x
            else:
                raise ValueError(f"Unsupported oversampling level: {oversampling}")
            
            # Generate projections for this method/oversampling combination
            test_projections = torch_projectors.project_2d_forw(
                image_to_use.unsqueeze(0),  # Add batch dimension
                rotations,
                shifts=None,
                output_shape=(base_size, base_size),
                interpolation=interp_method,
                oversampling=oversampling
            ).squeeze(0)  # Remove batch dimension
            
            # Calculate correlations against reference
            print(f"    Calculating correlations against reference...")
            
            # FSC in frequency space
            fsc_radii, fsc_correlations = calculate_fourier_shell_correlation(
                test_projections, reference_projections, 32
            )
            
            # Convert to real space for RSC
            test_real = torch.fft.irfft2(test_projections, s=(base_size, base_size))
            test_real = torch.fft.ifftshift(test_real, dim=(-2, -1))
            
            # RSC and intensity ratios
            max_radius = base_size // 2 - 1
            rsc_radii, rsc_correlations, intensity_ratios = calculate_real_shell_correlation(
                test_real, reference_real, max_radius
            )
            
            # Store results for this method/oversampling combination
            method_results[f"{oversampling}x"] = {
                "fourier_shell_correlation": {
                    "radii": fsc_radii.tolist(),
                    "correlations": fsc_correlations.tolist()
                },
                "real_shell_correlation": {
                    "radii": rsc_radii.tolist(),
                    "correlations": rsc_correlations.tolist(),
                    "intensity_ratios": intensity_ratios.tolist()
                }
            }
            
            # Store debug images for first rotation
            debug_images[f"first_rotation_{interp_method}_{oversampling}x"] = test_real[0].cpu().numpy().tolist()
        
        results[interp_method] = method_results
    
    # Combine results
    final_results = {
        "interpolation_methods": results,
        "parameters": {
            "base_size": base_size,
            "mask_diameter": mask_diameter,
            "soft_edge": soft_edge,
            "num_angles": num_angles,
            "interpolation_methods": interpolation_methods,
            "test_oversampling_levels": test_oversampling_levels,
            "reference_oversampling": reference_oversampling
        },
        "debug_images": debug_images
    }
    
    return final_results


def create_debug_image_plots(results: Dict[str, Any], platform_name: str, output_dir: Path) -> None:
    """Create plots showing the debug images.
    
    Args:
        results: Benchmark results dictionary
        platform_name: Platform identifier  
        output_dir: Output directory path
    """
    debug_images = results["debug_images"]
    interpolation_methods = results["parameters"]["interpolation_methods"]
    test_oversampling_levels = results["parameters"]["test_oversampling_levels"]
    
    # Convert back to numpy arrays
    original_checkerboard = np.array(debug_images["original_checkerboard"])
    filtered_checkerboard = np.array(debug_images["filtered_checkerboard"])
    reference_30x = np.array(debug_images["first_rotation_reference_30x"])
    
    # Get all test results dynamically
    test_images = {}
    for interp_method in interpolation_methods:
        test_images[interp_method] = {}
        for oversampling in test_oversampling_levels:
            key = f"first_rotation_{interp_method}_{oversampling}x"
            test_images[interp_method][oversampling] = np.array(debug_images[key])
    
    # Calculate number of rows needed: 1 for pipeline + 1 for each interpolation method
    num_rows = 1 + len(interpolation_methods)
    num_cols = max(3, len(test_oversampling_levels) + 1)  # At least 3 cols, or enough for all oversampling + 1 for diff
    
    # Create figure with dynamic grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
    
    # Ensure axes is 2D even for single row
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Top row: processing pipeline + reference
    im1 = axes[0,0].imshow(original_checkerboard, cmap='gray', origin='lower')
    axes[0,0].set_title('1. Original Checkerboard')
    plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
    
    im2 = axes[0,1].imshow(filtered_checkerboard, cmap='gray', origin='lower')
    axes[0,1].set_title('2. Low-pass Filtered')
    plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
    
    im3 = axes[0,2].imshow(reference_30x, cmap='gray', origin='lower')
    axes[0,2].set_title('3. Reference (30x Linear)')
    plt.colorbar(im3, ax=axes[0,2], shrink=0.8)
    
    # Hide unused axes in first row
    for col in range(3, num_cols):
        axes[0,col].set_visible(False)
    
    # Create rows for each interpolation method
    plot_idx = 4
    for row_idx, interp_method in enumerate(interpolation_methods, 1):
        # Plot each oversampling level for this interpolation method
        for col_idx, oversampling in enumerate(test_oversampling_levels):
            img = test_images[interp_method][oversampling]
            im = axes[row_idx, col_idx].imshow(img, cmap='gray', origin='lower')
            axes[row_idx, col_idx].set_title(f'{plot_idx}. {interp_method.title()} {oversampling}x')
            plt.colorbar(im, ax=axes[row_idx, col_idx], shrink=0.8)
            plot_idx += 1
        
        # Add difference plot in the last column (compare highest oversampling to reference)
        if len(test_oversampling_levels) > 0:
            highest_oversampling = max(test_oversampling_levels)
            diff_img = test_images[interp_method][highest_oversampling] - reference_30x
            diff_max = np.abs(diff_img).max()
            im_diff = axes[row_idx, len(test_oversampling_levels)].imshow(
                diff_img, cmap='RdBu', origin='lower', 
                vmin=-diff_max, vmax=diff_max
            )
            axes[row_idx, len(test_oversampling_levels)].set_title(f'{plot_idx}. {interp_method.title()} {highest_oversampling}x - Reference')
            plt.colorbar(im_diff, ax=axes[row_idx, len(test_oversampling_levels)], shrink=0.8)
            plot_idx += 1
        
        # Hide unused axes in this row
        for col in range(len(test_oversampling_levels) + 1, num_cols):
            axes[row_idx, col].set_visible(False)
    
    # Add axis labels and improve layout
    for ax in axes.flatten():
        if ax.get_visible():
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Pixels')
    
    fig.suptitle('Interpolation vs Reference Comparison', fontsize=16)
    plt.tight_layout()
    
    # Save debug images plot
    debug_plot_file = output_dir / f"{platform_name}_debug_images.png"
    plt.savefig(debug_plot_file, dpi=300, bbox_inches='tight')
    print(f"Debug images saved to: {debug_plot_file}")
    
    # Also save as PDF
    debug_pdf_file = output_dir / f"{platform_name}_debug_images.pdf"
    plt.savefig(debug_pdf_file, bbox_inches='tight')
    print(f"Debug images PDF saved to: {debug_pdf_file}")
    
    plt.close()


def save_results_and_plot(results: Dict[str, Any], platform_name: str) -> None:
    """Save results to JSON and create plot.
    
    Args:
        results: Benchmark results dictionary
        platform_name: Platform identifier
    """
    # Create output directories
    output_dir = Path(__file__).parent.parent / "data" / "interpolation_accuracy"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results["metadata"] = {
        "platform_name": platform_name,
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "torch_projectors_version": getattr(torch_projectors, '__version__', 'unknown')
    }
    
    # Save JSON results
    json_file = output_dir / f"{platform_name}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_file}")
    
    # Create combined correlation plots showing all 4 curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    interpolation_methods = results["parameters"]["interpolation_methods"]
    test_oversampling_levels = results["parameters"]["test_oversampling_levels"]
    
    # Define colors and styles for each combination
    curve_styles = {
        ('linear', 1.0): {'color': 'blue', 'linestyle': '-', 'marker': 'o'},
        ('linear', 1.5): {'color': 'blue', 'linestyle': '-.', 'marker': 'd'},
        ('linear', 2.0): {'color': 'blue', 'linestyle': '--', 'marker': 's'},
        ('cubic', 1.0): {'color': 'red', 'linestyle': '-', 'marker': '^'},
        ('cubic', 1.5): {'color': 'red', 'linestyle': '-.', 'marker': 'p'},
        ('cubic', 2.0): {'color': 'red', 'linestyle': '--', 'marker': 'v'}
    }
    
    # Plot FSC curves
    for interp_method in interpolation_methods:
        method_results = results["interpolation_methods"][interp_method]
        for oversampling in test_oversampling_levels:
            oversampling_key = f"{oversampling}x"
            style = curve_styles[(interp_method, oversampling)]
            
            fsc_data = method_results[oversampling_key]["fourier_shell_correlation"]
            if fsc_data["radii"] and fsc_data["correlations"]:
                ax1.plot(fsc_data["radii"], fsc_data["correlations"], 
                         color=style['color'], linestyle=style['linestyle'], 
                         marker=style['marker'], linewidth=2, markersize=6,
                         label=f'Bi-{interp_method.title()}, Oversampling = {oversampling}')
    
    ax1.set_xlabel('Normalized Frequency')
    ax1.set_ylabel('Fourier Ring Correlation')
    ax1.set_title('Fourier Ring Correlation vs Reference')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    
    # Plot intensity ratio curves
    for interp_method in interpolation_methods:
        method_results = results["interpolation_methods"][interp_method]
        for oversampling in test_oversampling_levels:
            oversampling_key = f"{oversampling}x"
            style = curve_styles[(interp_method, oversampling)]
            
            rsc_data = method_results[oversampling_key]["real_shell_correlation"]
            if rsc_data["radii"] and rsc_data["intensity_ratios"]:
                ax2.plot(rsc_data["radii"], rsc_data["intensity_ratios"], 
                         color=style['color'], linestyle=style['linestyle'], 
                         marker=style['marker'], linewidth=2, markersize=6,
                         label=f'Bi-{interp_method.title()}, Oversampling = {oversampling}')
    
    # Add theoretical bilinear filter response curve for linear interpolation
    if test_oversampling_levels and len(test_oversampling_levels) > 0:
        # Use any data to get radii for theoretical curve
        first_method = list(results["interpolation_methods"].keys())[0]
        first_oversampling = f"{test_oversampling_levels[0]}x"
        sample_radii = np.array(results["interpolation_methods"][first_method][first_oversampling]["real_shell_correlation"]["radii"])
        
        if len(sample_radii) > 0:
            base_size = results["parameters"]["base_size"]
            theoretical_response = calculate_bilinear_filter_response(sample_radii, base_size)
            ax2.plot(sample_radii, theoretical_response, 
                     color='black', linestyle=':', linewidth=2, 
                     label='Theoretical Bilinear Filter')
    
    ax2.set_xlabel('Real Space Ring Radius (pixels)')
    ax2.set_ylabel('Intensity RMS Attenuation (Test / Reference)')
    ax2.set_title('Real Space Intensity Attenuation vs Reference')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 31)
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    
    # Add overall title and parameter info
    params = results["parameters"]
    fig.suptitle('Interpolation Accuracy', fontsize=16)
    
    plt.tight_layout()
    
    # Save correlation plot
    plot_file = output_dir / f"{platform_name}_correlation.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Correlation plot saved to: {plot_file}")
    
    # Also save as PDF for paper
    pdf_file = output_dir / f"{platform_name}_correlation.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"Correlation PDF saved to: {pdf_file}")
    
    plt.close()
    
    # Create debug image plots
    create_debug_image_plots(results, platform_name, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Benchmark interpolation accuracy using real shell correlation")
    parser.add_argument('--platform-name', required=True,
                       help='Platform identifier (e.g., "a100-cuda", "m2-mps", "intel-cpu")')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda, mps (default: auto)')
    args = parser.parse_args()
    
    # Parse device
    device = parse_device(args.device)
    
    print(f"Starting interpolation accuracy benchmark")
    print(f"Platform: {args.platform_name}")
    print(f"Device: {device}")
    
    try:
        # Run the benchmark
        results = run_interpolation_benchmark(device)
        
        # Save results and create plot
        save_results_and_plot(results, args.platform_name)
        
        print(f"\nBenchmark completed successfully!")
        
        # Print statistics for each interpolation method and oversampling level
        for interp_method in results['parameters']['interpolation_methods']:
            method_results = results['interpolation_methods'][interp_method]
            
            print(f"\n{interp_method.upper()} INTERPOLATION RESULTS:")
            print(f"═" * 60)
            
            # Statistics for each oversampling level
            for oversampling in results['parameters']['test_oversampling_levels']:
                oversampling_key = f"{oversampling}x"
                level_results = method_results[oversampling_key]
                
                print(f"\n  {oversampling}x Oversampling vs Reference:")
                print(f"  " + "─" * 40)
                
                # FSC statistics
                fsc_corrs = level_results['fourier_shell_correlation']['correlations']
                if fsc_corrs:
                    print(f"  Fourier Ring Correlation (FRC):")
                    print(f"    Mean FRC: {np.mean(fsc_corrs):.4f}")
                    print(f"    Min FRC: {np.min(fsc_corrs):.4f}")
                    print(f"    Max FRC: {np.max(fsc_corrs):.4f}")
                
                # RSC statistics
                rsc_corrs = level_results['real_shell_correlation']['correlations']
                if rsc_corrs:
                    print(f"  Real Shell Correlation (RSC):")
                    print(f"    Mean RSC: {np.mean(rsc_corrs):.4f}")
                    print(f"    Min RSC: {np.min(rsc_corrs):.4f}")
                    print(f"    Max RSC: {np.max(rsc_corrs):.4f}")
                
                # Intensity ratio statistics
                intensity_ratios = level_results['real_shell_correlation']['intensity_ratios']
                if intensity_ratios:
                    print(f"  Intensity Ratio (Test / Reference):")
                    print(f"    Mean Ratio: {np.mean(intensity_ratios):.4f}")
                    print(f"    Min Ratio: {np.min(intensity_ratios):.4f}")
                    print(f"    Max Ratio: {np.max(intensity_ratios):.4f}")
                    print(f"    Ratio at Nyquist: {intensity_ratios[-1]:.4f}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()