# torch-projectors

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

The library provides two main functions:

- `forward_project_2d()`: Project 3D Fourier reconstructions to 2D projections
- `backward_project_2d()`: Backward projection for gradient computation

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

## Architecture

### Core Components

- **Python API**: `torch_projectors/ops.py` - Main user interface
- **C++ Kernels**: `csrc/cpu/cpu_kernels.cpp` - High-performance CPU implementations  
- **CUDA Kernels**: `csrc/cuda/cuda_kernels.cu` - GPU acceleration (when available)
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

## Development Status

This project is under active development. Current capabilities include 2D forward projection with full gradient support. The architecture is designed to support future expansion to 3D-to-2D and 3D-to-3D projections.