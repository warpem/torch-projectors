"""
Test utilities for torch-projectors testing.

This module contains shared utility functions used across multiple test files.
"""

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless testing
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest


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
        if tensor.is_complex():
            ax_real.imshow(tensor.real.detach().numpy(), cmap='viridis', origin='lower')
            ax_real.set_title(f"{title} (Real)")
        else:
            ax_real.imshow(tensor.detach().numpy(), cmap='viridis', origin='lower')
            ax_real.set_title(f"{title}")
        ax_real.axis('off')

        # Plot imaginary part
        ax_imag = axes[2 * row + 1, col]
        if tensor.is_complex():
            ax_imag.imshow(tensor.imag.detach().numpy(), cmap='viridis', origin='lower')
            ax_imag.set_title(f"{title} (Imag)")
        else:
            # For real tensors, leave the imaginary plot empty
            ax_imag.text(0.5, 0.5, f"{title}\n(Real tensor)", ha='center', va='center', 
                        transform=ax_imag.transAxes, fontsize=12)
            ax_imag.set_title(f"{title} (N/A)")
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


def create_fourier_mask(shape, radius_cutoff_sq, device=None):
    """
    Creates a boolean mask for a Fourier-space tensor, zeroing out frequencies
    outside a specified radius.
    """
    H, W_half = shape[-2], shape[-1]
    ky = torch.arange(H, dtype=torch.float32, device=device)
    ky[ky > H // 2] -= H
    kx = torch.arange(W_half, dtype=torch.float32, device=device)
    kyy, kxx = torch.meshgrid(ky, kx, indexing='ij')
    
    return kxx**2 + kyy**2 >= radius_cutoff_sq


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
    ], device=matrix_grad.device)
    
    # Angle gradient = trace(matrix_grad^T * dR_dtheta)
    return torch.sum(matrix_grad * dR_dtheta)


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


def complex_mse_loss(input_tensor, target_tensor):
    """Manual implementation of MSE loss for complex tensors."""
    diff = input_tensor - target_tensor
    # |a + ib|^2 = a^2 + b^2
    loss = torch.mean(diff.real**2 + diff.imag**2)
    return loss


def create_friedel_symmetric_noise(shape, device=None):
    """
    Create complex noise that satisfies Friedel symmetry for RFFT format.
    
    For real-valued images, the Fourier transform must satisfy:
    F(-k) = F*(k) (complex conjugate)
    
    Works for both 2D and 3D cases:
    - 2D: shape = (H, W) where W is RFFT width (original_width//2 + 1)
    - 3D: shape = (D, H, W) where W is RFFT width (original_width//2 + 1)
    
    Args:
        shape: tuple of dimensions
        device: torch device
    
    Returns:
        Complex tensor with proper Friedel symmetry
    """
    noise = torch.randn(*shape, dtype=torch.complex64, device=device)
    
    if len(shape) == 2:
        # 2D case: (H, W)
        H, W = shape
        
        # DC component must be real
        noise[0, 0] = torch.complex(noise[0, 0].real, torch.tensor(0.0, device=device))
        
        # Nyquist frequency (if H is even) must be real
        if H % 2 == 0:
            noise[H//2, 0] = torch.complex(noise[H//2, 0].real, torch.tensor(0.0, device=device))
        
        # Friedel symmetry on kx=0 line: F(ky, 0) = F*(-ky, 0)
        for ky in range(1, H//2):
            noise[H-ky, 0] = torch.conj(noise[ky, 0])
            
    elif len(shape) == 3:
        # 3D case: (D, H, W)
        D, H, W = shape
        
        # DC component must be real
        noise[0, 0, 0] = torch.complex(noise[0, 0, 0].real, torch.tensor(0.0, device=device))
        
        # Nyquist frequencies must be real
        if D % 2 == 0:
            noise[D//2, 0, 0] = torch.complex(noise[D//2, 0, 0].real, torch.tensor(0.0, device=device))
        if H % 2 == 0:
            noise[0, H//2, 0] = torch.complex(noise[0, H//2, 0].real, torch.tensor(0.0, device=device))
        if D % 2 == 0 and H % 2 == 0:
            noise[D//2, H//2, 0] = torch.complex(noise[D//2, H//2, 0].real, torch.tensor(0.0, device=device))
        
        # Friedel symmetry on kx=0 plane: F(kz, ky, 0) = F*(-kz, -ky, 0)
        for kz in range(D):
            for ky in range(H):
                if kz == 0 and ky == 0:
                    continue  # Already handled DC
                
                # Calculate symmetric indices
                kz_sym = (-kz) % D
                ky_sym = (-ky) % H
                
                # Skip if we're at the positive frequency (we set the negative one)
                if kz > D//2 or (kz == D//2 and ky > H//2):
                    continue
                if kz == 0 and ky > H//2:
                    continue
                    
                # Set symmetric counterpart
                noise[kz_sym, ky_sym, 0] = torch.conj(noise[kz, ky, 0])
    
    else:
        raise ValueError(f"Unsupported shape dimensionality: {len(shape)}. Only 2D and 3D supported.")
    
    return noise


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Test fixture that yields available devices"""
    device_type = request.param
    if device_type == "mps":
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this system")
        if not torch.backends.mps.is_built():
            pytest.skip("MPS not built with this PyTorch installation")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available on this system")
            
    return torch.device(device_type)