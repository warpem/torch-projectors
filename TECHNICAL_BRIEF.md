RAW INITIAL PROMPT:
we want to develop a library of pytorch operators to perform differentiable forward and backward projection in cryo-em and related fields. in the long run, we want to support cpu, cuda, and mps backends, but we'll start with cpu for now. we want to support linux, mac, and windows platforms, but we'll start developing this on a mac system for now. we want to target pytorch 2.4+ and make use of its improved api for externally defined operators. we also want to provide c api bindings for developers using libtorch from non-python languages. we want to distribute the library as a collection of pip package wheels (for each config matrix permutation).
@research.md contains some findings we gathered about the current state of pytorch operator development.
here is what we want to implement:
for now, we'll focus on projection in fourier space.
2d reconstruction -> forward projection -> 2d image
2d image -> backward projection -> 2d reconstruction
3d reconstruction -> forward projection -> 2d image
2d image -> backward projection -> 3d reconstruction
3d reconstruction -> forward projection -> 3d volume
3d volume -> backward projection -> 3d reconstruction

as you know, 2d projections are central slices through the fourier transform of a 3d volume.

we want to support nearest neighbor, bi/tri-linear, and bi/tri-cubic sampling in both directions.

we want to support oversampling to make high-quality interpolation computationally cheaper at the expense of memory. e.g. we zero-pad a reconstruction in real-space by 2x, and when we project it we evaluate a grid of the original box size but scale the evaluated coordinates up 2x. same in the other direction: we insert the ft components of the original image or volume at 2x upscaled coordinates into the reconstruction volume, filling it sparsely, then crop it in real-space to arrive at the final reconstruction. this way we can get away with linear interpolation at 2x or 3x oversampling in many cases.

we want to support sampling a smaller grid than the reconstruction's box size, effectively producing a downsampled projection with a larger pixel size. we also want to support only sampling components up to a certain radius in fourier-space, allowing the instant application of a top-hat low-pass filter.

we want to support a batch dimension to enable the simultaneous (back)projection of/into multiple reconstructions. we want to follow pytorch's standard rules here. each reconstruction will only have 1 channel, so no need to support the channel dimension.

all data will be in the standard fftw format, with its center at element 0. we need to consider this every time we sample from/into it. some rotations will also cause some components or some of their sampling kernel's support to come into the -x region where the friedel symmetry applies (our reconstructions are real-numbered in real-space, so the ft is only width/2+1 in width). we need to apply the friedel symmetry transparently during sampling.

we don't care about determinism in this package, so it's ok to use atomic operations to insert data without pre-sorting anything.

for the gradient, we need to consider that at x=0 we have both the regular and friedel-symmetric components, which is an aberration compared to every component at x>0. if we consider both members of those pairs in gradient calculation, they will have twice the gradient as the x>0 components.

rotations will be provided as 3x3 matrices. shifts will be provided as 2- or 3-tuples depending on data dimensionality. the rotations are defined such that after transforming the grid coordinates by the matrix, that's where we want to sample the reconstruction. shifts are defined in the reference frame of the projection data, i.e. an image is first shifted to align its rotation center with the center, then it's rotated. if the user requests it, we want to provide gradients for the matrix components as well as the shifts (assuming shifts are performed through phase modulation in fourier-space).

the project should be well-structured, with compilation and testing set up.

# Technical Brief & Specification: torch-projectors

## 1. Vision & Scope

**Project Vision:** To create a high-performance, differentiable 2D and 3D projection library for PyTorch, tailored for applications in cryogenic electron microscopy (cryo-EM), tomography, and related fields.

**Core Goal:** The library, `torch-projectors`, will provide efficient and accurate operators for forward projection (simulating images from reconstructions) and backward projection (reconstructing volumes from images). A key focus is on differentiability, enabling its use in gradient-based optimization algorithms for structure refinement and solving inverse problems.

**Target Ecosystem:**
*   **PyTorch Version:** 2.4+
*   **Platforms:** Linux, macOS, Windows
*   **Backends:** CPU (initial focus), with future support for CUDA and Apple MPS.
*   **API:** Pythonic API with seamless PyTorch integration, and C++ bindings for the LibTorch ecosystem.
*   **Distribution:** Pre-compiled `pip` wheels for a wide matrix of platforms, Python versions, and hardware backends.

## 2. Core Functionality & Operator Design

The library will be centered around two main operators: `forward_project` and `backward_project`. These operators perform sampling in Fourier space, corresponding to the Projection-Slice Theorem. Forward projection is a "gather" operation, while backward projection is a "scatter" operation.

### 2.1. Operator Signatures (Python API)

```python
# torch_projectors/ops.py

import torch

def forward_project(
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts: torch.Tensor | None = None,
    output_shape: tuple[int, ...],
    interpolation: str = 'linear',
    oversampling: int = 1,
    fourier_radius_cutoff: float | None = None
) -> torch.Tensor:
    """
    Projects a 2D or 3D reconstruction into 2D images or 3D volumes.

    This operation corresponds to sampling a Fourier-space volume at specified
    grid locations defined by rotations and shifts.
    """
    ...

def backward_project(
    projections: torch.Tensor,
    rotations: torch.Tensor,
    shifts: torch.Tensor | None = None,
    reconstruction_shape: tuple[int, ...],
    interpolation: str = 'linear',
    oversampling: int = 1
) -> torch.Tensor:
    """
    Back-projects 2D images or 3D volumes into a 2D or 3D reconstruction.

    This operation corresponds to inserting (scattering) projection data into a
    Fourier-space volume at locations defined by rotations and shifts.
    """
    ...
```

### 2.2. Supported Projection Geometries

The operators will infer dimensionality from the input tensors:

*   **3D -> 2D:** Project a 3D reconstruction to 2D images (e.g., single-particle cryo-EM).
    *   `reconstruction`: `(B, D, H, W//2+1)`
    *   `projections`: `(B, H', W')`
*   **3D -> 3D:** Resample a 3D reconstruction into a new 3D grid (e.g., tomogram simulation).
    *   `reconstruction`: `(B, D, H, W//2+1)`
    *   `projections`: `(B, D', H', W')`
*   **2D -> 2D:** Rotate and resample a 2D image (e.g., 2D image alignment).
    *   `reconstruction`: `(B, H, W//2+1)`
    *   `projections`: `(B, H', W')`

### 2.3. C++ API (LibTorch)

By using the `TORCH_LIBRARY` mechanism for registration, the same operators will be exposed to C++ under the `torch_projectors` namespace.

```cpp
// Example C++ usage
#include <torch/torch.h>

// After loading the library
auto projected = torch::ops::torch_projectors::forward_project(
    reconstruction, rotations, shifts, ...
);
```

## 3. Technical Implementation Details

### 3.1. Fourier Space Data and Coordinates

*   **Data Format:** All reconstructions and projections in Fourier space will use the standard real-to-complex (RFFT) format, as produced by `torch.fft.rfftn`. The last dimension will be `N/2 + 1`.
*   **Coordinate System:** The origin `(0,0,0)` is at index `[..., 0, 0, 0]`. Fourier coordinates for sampling range from `-(N-1)//2` to `N//2`. The implementation must correctly map these conceptual coordinates to array indices.

### 3.2. Friedel Symmetry

Since reconstructions are real-valued, their Fourier transforms exhibit Friedel symmetry: `F(k) = conj(F(-k))`. This must be handled transparently during sampling when a transformed coordinate falls into the half-plane not explicitly stored by the RFFT. The kernel will compute the value by fetching the complex conjugate from the corresponding symmetric location.

### 3.3. Interpolation Methods

The following interpolation methods will be supported for both forward and backward projection:
*   `'nearest'`: Nearest neighbor sampling.
*   `'linear'`: Bi-linear for 2D and Tri-linear for 3D.
*   `'cubic'`: Bi-cubic for 2D and Tri-cubic for 3D.

### 3.4. Differentiability and Gradients

This is a critical feature of the library.
*   **Operator Gradients:** The backward pass for `forward_project` is a `backward_project` operation on the output gradient, and vice versa. The implementation must correctly transpose the interpolation logic.
*   **Pose Gradients:** Gradients with respect to `rotations` and `shifts` will be supported.
    *   **Shifts:** Implemented as phase modulation in Fourier space. The gradient is straightforward to compute analytically.
    *   **Rotations:** Gradients are computed via the chain rule, requiring the derivative of the interpolation kernel with respect to sample coordinates.
*   **Friedel Symmetry in Gradients:** Special care will be taken for DC (`k=0`) and Nyquist components to ensure gradients are not double-counted, as these components are their own symmetric pairs.

### 3.5. Advanced Sampling Features

*   **Oversampling:** An integer factor (`oversampling > 1`) will simulate sampling from a reconstruction that was zero-padded in real space. This is achieved by scaling the sampling coordinates, providing a computationally cheaper path to higher-quality interpolation.
*   **Downsampling & Filtering:**
    *   Specifying an `output_shape` smaller than the reconstruction implies downsampling.
    *   The `fourier_radius_cutoff` parameter will discard samples beyond a specified frequency, acting as a top-hat low-pass filter.

### 3.6. Performance

*   **Atomics:** For `backward_project`, atomic operations (`atomicAdd`) will be used on CUDA and supported C++ atomics for CPU to enable efficient, parallel insertion of projection data into the reconstruction volume without requiring sorting or synchronization. This makes the operation non-deterministic, which is acceptable for the target applications.

## 4. Project Architecture & Structure

The project will follow the officially recommended hybrid C++/Python extension pattern for PyTorch 2.4+ to ensure high performance, stability, and seamless integration with PyTorch's subsystems.

*   **C++ Forward and Backward Kernels:** The core computational logic for both the forward and backward passes will be implemented as separate C++ functions for maximum performance.
*   **`TORCH_LIBRARY` Operator Registration:** Both the forward and backward C++ kernels will be registered as distinct PyTorch operators using the `TORCH_LIBRARY` API. This makes the high-performance C++ functions callable directly from Python.
*   **Python Autograd Linkage:** In Python, we will use `torch.library.register_autograd` to link the C++ forward and backward operators. This crucial step tells PyTorch's autograd engine that the gradient for our `forward_project` operator is computed by calling our `backward_project` operator. This pattern is the modern, stable, and recommended approach.

### 4.1. Directory Structure

```
torch-projectors/
├── setup.py
├── torch_projectors/
│   ├── __init__.py
│   └── ops.py          # Python operator definitions and registration
├── csrc/
│   ├── torch_projectors.cpp    # TORCH_LIBRARY definitions
│   ├── cpu/
│   │   ├── cpu_kernels.cpp     # CPU kernel implementations
│   │   └── cpu_kernels.h
│   └── cuda/
│       ├── cuda_kernels.cu     # CUDA kernel implementations
│       └── cuda_kernels.h
└── tests/
    └── test_ops.py             # Pytest tests
```

### 4.2. Build System

*   **`setuptools` & `torch.utils.cpp_extension`:** The primary build system will use `setuptools`, integrating with PyTorch's `CUDAExtension` and `CppExtension` to handle the C++/CUDA compilation. This is the most direct path for Python-centric libraries.
*   **`pyproject.toml`:** A `pyproject.toml` file is crucial for modern `setuptools` builds. It specifies build-time dependencies, such as `torch`, preventing `ModuleNotFoundError` during the isolated build process that `pip` uses.
*   **CMake (Potential Future):** If the C++ logic becomes highly complex or requires dependencies outside the PyTorch ecosystem, a transition to a CMake-based build could be considered.

### 4.3. Testing Strategy

*   **Unit Tests:** `pytest` will be used for all Python-level testing.
*   **`torch.library.opcheck`:** We will leverage PyTorch 2.4's `opcheck` utility for comprehensive validation of the custom operators, including correctness, autograd, and meta tensor registration.
*   **Numerical Precision:** Tests will compare kernel outputs against a reference implementation (e.g., `torch.nn.functional.grid_sample`) where applicable, accounting for floating-point tolerances.
*   **CI/CD:** GitHub Actions will be configured to run tests across the full build matrix (OS, Python version, backend).

## 5. Distribution Strategy

*   **PyPI Wheels:** The primary distribution method will be pre-compiled binary wheels on PyPI.
*   **Build Matrix:** A CI/CD pipeline (using `cibuildwheel`) will build and publish wheels for all supported combinations of:
    *   Operating System: `manylinux`, `windows`, `macos`
    *   Python Version: 3.9, 3.10, 3.11, 3.12
    *   CPU Architecture: `x86_64`, `arm64` (Apple Silicon)
    *   CUDA Version (for CUDA-enabled wheels)
*   **Version Naming:** Wheel versions will be named to indicate the CUDA version they are built against (e.g., `1.0.0+cu121`).

## 6. Development Roadmap & Milestones

### Milestone 1: CPU Backend Foundation
1.  **Project Scaffolding:** Set up the repository with the defined structure, `setup.py`, and initial CI configuration.
2.  **Operator Registration:** Define the `forward_project` and `backward_project` operators in `torch_projectors.cpp` using `TORCH_LIBRARY`.
3.  **CPU Kernel (3D -> 2D):** Implement the core logic for the 3D-to-2D forward projection on the CPU, starting with linear interpolation.
4.  **Initial Tests:** Write initial `opcheck` and numerical correctness tests for the CPU forward projection.
5.  **CPU Backward Projection:** Implement the CPU kernel for backward projection.
6.  **Full CPU Feature Set:** Add support for other geometries, interpolation methods, and advanced sampling features on the CPU backend.
7.  **Autograd:** Implement the `autograd` formulas for all CPU operations, including pose gradients.

### Milestone 2: CUDA Backend
1.  Extend the build system to compile CUDA sources.
2.  Implement CUDA kernels for all projection functionalities.
3.  Optimize CUDA kernels for performance (e.g., shared memory usage).
4.  Expand CI to build and test CUDA-enabled wheels.

### Milestone 3: MPS Backend & Broader Distribution
1.  Investigate and implement Metal Performance Shaders (MPS) kernels for Apple Silicon GPUs.
2.  Finalize the `cibuildwheel` matrix for comprehensive wheel publication.
3.  Publish the first official version to PyPI. 