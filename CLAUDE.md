# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`torch-projectors` is a high-performance, differentiable 2D and 3D projection library for PyTorch, designed for cryogenic electron microscopy (cryo-EM) and tomography applications. The library provides forward and backward projection operators that work in Fourier space, following the Projection-Slice Theorem.

## Development Environment Setup

This project requires a conda environment with PyTorch and pytest:

```bash
# Create and activate environment
conda create -n torch-projectors python=3.11 -y
conda activate torch-projectors
conda install pytorch pytest -c pytorch -c conda-forge -y

# Install in editable mode (compiles C++ extension)
python -m pip install -e .
```

## Core Architecture

The project uses PyTorch's modern hybrid C++/Python extension pattern:

- **C++ Kernels**: High-performance forward/backward operations in `csrc/cpu/cpu_kernels.cpp`
- **TORCH_LIBRARY Registration**: Operators registered via `TORCH_LIBRARY` in `csrc/torch_projectors.cpp`
- **Python Autograd**: `torch.library.register_autograd` links C++ operators in `torch_projectors/ops.py`

### Key Components

- `torch_projectors/ops.py`: Python API with `forward_project_2d()` and `backward_project_2d()`
- `csrc/torch_projectors.cpp`: Operator registration defining the `torch_projectors` namespace
- `csrc/cpu/cpu_kernels.{h,cpp}`: CPU implementations of projection kernels
- `torch_projectors/__init__.py`: Library loading and Python registration

## Data Format and Coordinate System

- **Fourier Space Data**: Uses PyTorch's RFFT format (last dimension is `N/2 + 1`)
- **Coordinate Origin**: `(0,0,0)` is at index `[..., 0, 0, 0]`
- **Friedel Symmetry**: Automatically handled for real-valued reconstructions
- **Batch Dimensions**: Two batch dimensions - first for reconstructions, second for poses

## Development Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ops.py

# Run specific test function
pytest tests/test_ops.py::test_function_name

# Install in editable mode (rebuild C++ extension)
python -m pip install -e .
```

## Testing

Tests use pytest and generate visualization outputs in `test_outputs/`. The test suite includes:
- Numerical correctness validation
- Gradient checking via autograd
- Visual validation with matplotlib plots
- Batch processing verification

## Key Features

- **Interpolation Methods**: Nearest neighbor, linear, and cubic interpolation
- **Oversampling**: Computational efficiency through coordinate scaling
- **Fourier Filtering**: Optional radius cutoff for low-pass filtering
- **Differentiability**: Full gradient support for reconstructions, rotations, and shifts
- **Batch Processing**: Efficient handling of multiple reconstructions and poses

## Future Expansion

The architecture is designed to support:
- CUDA backend (kernels in `csrc/cuda/`)
- MPS backend for Apple Silicon
- 3D-to-3D and other projection geometries
- C++ API for LibTorch integration