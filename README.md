# torch-projectors

![Mac: CPU, MPS](https://github.com/warpem/torch-projectors/actions/workflows/test-mac.yml/badge.svg)
![Windows: CPU](https://github.com/warpem/torch-projectors/actions/workflows/test-windows.yml/badge.svg)
![Linux: CPU, CUDA](https://github.com/warpem/torch-projectors/actions/workflows/test-linux.yml/badge.svg)

A high-performance, differentiable 2D and 3D projection library for PyTorch, designed for cryogenic electron microscopy (cryo-EM) and tomography applications. The library provides forward and backward projection operators that work in Fourier space, following the Projection-Slice Theorem.

## Features

- **Multi-Platform Support**: CPU, CUDA (when available), and Metal Performance Shaders (MPS) on Apple Silicon
- **Multiple Backends**: Optimized kernels for different hardware platforms
- **Interpolation Methods**: Nearest neighbor, linear, and cubic interpolation
- **Fourier Space Operations**: Efficient projections using PyTorch's RFFT format
- **Full Differentiability**: Gradient support for reconstructions, rotations, and shifts
- **Batch Processing**: Efficient handling of multiple reconstructions and poses
- **Oversampling Support**: Computational efficiency through coordinate scaling
- **Fourier Filtering**: Optional radius cutoff for low-pass filtering

## Core API

The library provides four main high-level functions:

### 2D-to-2D Operations
- `project_2d_forw()`: Forward project 2D Fourier reconstructions to 2D projections
- `backproject_2d_forw()`: Back-project 2D projections into 2D reconstructions (adjoint operation)

### 3D-to-2D Operations  
- `project_3d_to_2d_forw()`: Forward project 3D Fourier volumes to 2D projections

### 2D-to-3D Operations
- `backproject_2d_to_3d_forw()`: Back-project 2D projections into 3D reconstructions (adjoint operation)

## Installation & Development Setup

### Prerequisites

This project requires a conda environment with PyTorch and pytest:

```bash
# Create and activate environment
conda create -n torch-projectors python=3.11 -y
conda activate torch-projectors
conda install pytorch pytest -c pytorch -c conda-forge -y
```

### Install in Editable Mode

The project uses PyTorch's modern hybrid C++/Python extension pattern with automatic platform detection:

```bash
# Install the package (compiles C++ extensions automatically)
python -m pip install -e .
```

The build system automatically detects and enables:
- **CUDA support** when CUDA is available and `TORCH_CUDA_ARCH_LIST` is set
- **MPS support** on macOS with Apple Silicon
- **CPU fallback** on all platforms

## Usage Examples

This section demonstrates minimal usage patterns for the main projection operations with oversampling:

### 2D-to-2D Forward Projection

```python
import torch
import torch_projectors

# Helper function to pad and prepare real-space data
def pad_and_fftshift(tensor, oversampling_factor):
    H, W = tensor.shape[-2:]
    new_size = int(H * oversampling_factor)
    if new_size % 2 != 0:
        new_size += 1
    pad_total = new_size - H
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    padded = torch.nn.functional.pad(tensor, (pad_before, pad_after, pad_before, pad_after))
    return torch.fft.fftshift(padded, dim=(-2, -1))

# Start with real-space image
real_image = torch.randn(32, 32)

# 1. Zero pad 2x and fftshift
padded_image = pad_and_fftshift(real_image, 2.0)

# 2. Convert to Fourier space
fourier_image = torch.fft.rfft2(padded_image, norm='forward')

# 3. Set up projection parameters (90-degree rotation)
rotations = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
shifts = torch.zeros(1, 1, 2, dtype=torch.float32)

# 4. Forward project with oversampling=2.0
projection = torch_projectors.project_2d_forw(
    fourier_image.unsqueeze(0),  # Add batch dimension
    rotations,
    shifts=shifts,
    output_shape=(32, 32),
    interpolation='linear',
    oversampling=2.0
)

# 5. Convert back to real space
result = torch.fft.irfft2(projection[0, 0], s=(32, 32))
result = torch.fft.ifftshift(result)
```

### 2D-to-2D Backward Projection

```python
import torch
import torch_projectors

# Helper function to crop and ifftshift real-space data
def ifftshift_and_crop(real_tensor, oversampling_factor):
    shifted = torch.fft.ifftshift(real_tensor, dim=(-2, -1))
    current_size = real_tensor.shape[-1]
    original_size = int(current_size / oversampling_factor)
    crop_total = current_size - original_size
    crop_start = crop_total // 2
    crop_end = crop_start + original_size
    return shifted[..., crop_start:crop_end, crop_start:crop_end]

# Start with real-space image (e.g., a projection to backproject)
real_projection = torch.randn(32, 32)

# 1. fftshift and convert to Fourier space
shifted_projection = torch.fft.fftshift(real_projection)
fourier_projection = torch.fft.rfft2(shifted_projection, norm='forward')

# 2. Set up backprojection parameters
rotations = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
shifts = torch.zeros(1, 1, 2, dtype=torch.float32)

# 3. Backward project with oversampling=2.0
data_rec, weight_rec = torch_projectors.backproject_2d_forw(
    fourier_projection.unsqueeze(0).unsqueeze(0),  # Add batch and pose dimensions
    rotations,
    shifts=shifts,
    interpolation='linear',
    oversampling=2.0
)

# 4. Convert reconstruction to real space
real_reconstruction = torch.fft.irfft2(data_rec[0], norm='forward')

# 5. ifftshift and crop to 0.5x size (original size from 2x oversampling)
result = ifftshift_and_crop(real_reconstruction, 2.0)
```

### 3D-to-2D Forward Projection

```python
import torch
import torch_projectors

# Helper function to pad 3D volumes
def pad_and_fftshift_3d(tensor, oversampling_factor):
    D, H, W = tensor.shape[-3:]
    new_size = int(D * oversampling_factor)
    if new_size % 2 != 0:
        new_size += 1
    pad_total = new_size - D
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    padded = torch.nn.functional.pad(tensor, 
                                    (pad_before, pad_after,    # W
                                     pad_before, pad_after,    # H  
                                     pad_before, pad_after))   # D
    return torch.fft.fftshift(padded, dim=(-3, -2, -1))

# Start with 3D real-space volume
real_volume = torch.randn(32, 32, 32)

# 1. Zero pad 2x and fftshift
padded_volume = pad_and_fftshift_3d(real_volume, 2.0)

# 2. Convert to Fourier space
fourier_volume = torch.fft.rfftn(padded_volume, dim=(-3, -2, -1), norm='forward')

# 3. Set up projection parameters (90-degree rotation around Y axis)
rotations = torch.tensor([
    [0., 0., 1.],    # x' = z
    [0., 1., 0.],    # y' = y  
    [-1., 0., 0.]    # z' = -x
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
shifts = torch.zeros(1, 1, 2, dtype=torch.float32)

# 4. Forward project 3D->2D with oversampling=2.0
projection = torch_projectors.project_3d_to_2d_forw(
    fourier_volume.unsqueeze(0),  # Add batch dimension
    rotations,
    shifts=shifts,
    output_shape=(32, 32),
    interpolation='linear',
    oversampling=2.0
)

# 5. Convert back to real space
result = torch.fft.irfft2(projection[0, 0], s=(32, 32))
result = torch.fft.ifftshift(result)
```

### 2D-to-3D Backward Projection

```python
import torch
import torch_projectors

# Helper function to crop 3D volumes
def ifftshift_and_crop_3d(real_tensor, oversampling_factor):
    shifted = torch.fft.ifftshift(real_tensor, dim=(-3, -2, -1))
    current_size = real_tensor.shape[-3]
    original_size = int(current_size / oversampling_factor)
    crop_total = current_size - original_size
    crop_start = crop_total // 2
    crop_end = crop_start + original_size
    return shifted[..., crop_start:crop_end, crop_start:crop_end, crop_start:crop_end]

# Start with 2D real-space projection
real_projection = torch.randn(32, 32)

# 1. fftshift and convert to Fourier space
shifted_projection = torch.fft.fftshift(real_projection)
fourier_projection = torch.fft.rfft2(shifted_projection, norm='forward')

# 2. Set up backprojection parameters (rotation matrix for 3D)
rotations = torch.tensor([
    [1., 0., 0.],
    [0., 1., 0.], 
    [0., 0., 1.]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
shifts = torch.zeros(1, 1, 2, dtype=torch.float32)

# 3. Backward project 2D->3D with oversampling=2.0
data_rec, weight_rec = torch_projectors.backproject_2d_to_3d_forw(
    fourier_projection.unsqueeze(0).unsqueeze(0),  # Add batch and pose dimensions
    rotations,
    shifts=shifts,
    interpolation='linear',
    oversampling=2.0
)

# 4. Convert reconstruction to real space
real_reconstruction = torch.fft.irfftn(data_rec[0], dim=(-3, -2, -1), norm='forward')

# 5. ifftshift and crop to 0.5x size (original size from 2x oversampling)
result = ifftshift_and_crop_3d(real_reconstruction, 2.0)
```

## Architecture

### Core Components

- **Python API**: `torch_projectors/ops.py` - Main user interface
- **C++ Kernels**: 
  - `csrc/cpu/2d/projection_2d_kernels.cpp` - 2D forward/backward projection
  - `csrc/cpu/2d/backprojection_2d_kernels.cpp` - 2D back-projection (adjoint)
  - `csrc/cpu/3d/projection_3d_to_2d_kernels.cpp` - 3D-to-2D projection
  - `csrc/cpu/3d/backprojection_2d_to_3d_kernels.cpp` - 2D-to-3D back-projection (adjoint)
- **CUDA Kernels**: `csrc/cuda/*.cu` - GPU acceleration (when available)
- **Metal Shaders**: `csrc/mps/*.metal` - Apple Silicon optimization
- **Operator Registration**: `csrc/torch_projectors.cpp` - PyTorch integration

### Design Pattern

- **C++ Kernels**: Performance-critical forward/backward operations
- **TORCH_LIBRARY Registration**: Operators registered in the `torch_projectors` namespace
- **Python Autograd**: `torch.library.register_autograd` links C++ operators for seamless differentiation

## Data Format

- **Fourier Space**: Uses PyTorch's RFFT format (last dimension is `N/2 + 1`)
- **Coordinate System**: Origin `(0,0,0)` at index `[..., 0, 0, 0]`
- **Batch Dimensions**: Two batch dimensions - first for reconstructions, second for poses
- **Friedel Symmetry**: Automatically handled for real-valued reconstructions

## Testing

Comprehensive test suite with visual validation:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_basic_projection.py      # Core functionality
pytest tests/test_gradients.py            # Gradient verification
pytest tests/test_cross_platform.py       # Multi-platform consistency
pytest tests/test_performance.py          # Performance benchmarks
pytest tests/test_visual_validation.py    # Visual output validation
```

Tests generate visualization outputs in `test_outputs/` for manual inspection and include:
- Numerical correctness validation
- Gradient checking via autograd  
- Visual validation with matplotlib plots
- Cross-platform consistency verification
- Performance benchmarking

## Key Features

### 2D Back-Projection (New!)
- **Adjoint Operations**: Mathematical transpose of forward projection
- **Weight Accumulation**: Support for CTF² or other weight functions
- **Full Differentiability**: Gradients w.r.t. projections, weights, rotations, and shifts
- **Conjugate Phase Shifts**: Proper mathematical adjoint with conjugate phase corrections
- **Wiener Filtering Ready**: Separate data/weight accumulation enables downstream filtering

### Interpolation & Filtering
- **Interpolation Methods**: Linear (bilinear/trilinear) and cubic (bicubic/tricubic)
- **Oversampling Support**: Coordinate scaling for computational efficiency  
- **Fourier Filtering**: Optional radius cutoff for low-pass filtering
- **Friedel Symmetry**: Automatic handling for real-valued reconstructions

## Development Status

This project is under active development. Current capabilities include:
- ✅ 2D-to-2D forward projection with full gradient support
- ✅ 2D-to-2D back-projection (adjoint) with weight accumulation
- ✅ 3D-to-2D forward projection with full gradient support
- ✅ 2D-to-3D back-projection (adjoint) with full gradient support
- 🚧 3D-to-3D projection operations
- 🚧 3D-to-3D back-projection (adjoint) with weight accumulation
- 🚧 CUDA and MPS backend implementations

The architecture is designed to support future expansion to additional projection geometries and backend optimizations.
