# Creating PyTorch Packages with Custom CUDA Kernels: A Complete Guide

PyTorch 2.4's major API overhaul has fundamentally changed how developers create custom operators with CUDA kernels. The new high-level APIs dramatically simplify development while providing better integration with torch.compile and other PyTorch subsystems. This guide covers the complete workflow from implementation to distribution, incorporating the latest best practices and lessons learned from successful packages.

## Modern operator registration simplifies custom CUDA development

**PyTorch 2.4 introduced the `torch.library.custom_op()` API**, replacing complex low-level approaches with a streamlined registration system that automatically handles schema inference and provides guaranteed compatibility with PyTorch's modern subsystems. This represents the most significant change in custom operator development since PyTorch's inception.

The new approach eliminates the need for manual pybind11 bindings and complex torch.library APIs for basic use cases. Instead, developers can focus on kernel implementation while PyTorch handles the integration complexity. This shift enables better composability with torch.compile, automatic differentiation, and distributed training systems.

## Technical Implementation Architecture

### Project structure and build system setup

The recommended project structure separates operator definitions, implementations, and Python bindings:

```
my_extension/
├── setup.py
├── my_extension/
│   ├── __init__.py
│   └── ops.py
├── csrc/
│   ├── extension.cpp      # Operator definitions
│   ├── cpu_kernels.cpp    # CPU implementations
│   ├── cuda_kernels.cu    # CUDA kernel implementations
│   └── cuda_kernels.cpp   # CUDA interface layer
```

The build system uses PyTorch's `CUDAExtension` with setuptools:

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_extension",
    ext_modules=[
        CUDAExtension(
            name="my_extension._C",
            sources=[
                "csrc/extension.cpp",
                "csrc/cpu_kernels.cpp", 
                "csrc/cuda_kernels.cpp",
                "csrc/cuda_kernels.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
```

### Operator registration with modern APIs

The current best practice uses `TORCH_LIBRARY` macros for C++ operator definitions:

```cpp
// Define operator schema
TORCH_LIBRARY(my_extension, m) {
    m.def("my_op(Tensor input, Tensor other, float alpha) -> Tensor");
}

// Register CPU implementation
TORCH_LIBRARY_IMPL(my_extension, CPU, m) {
    m.impl("my_op", &my_op_cpu);
}

// Register CUDA implementation
TORCH_LIBRARY_IMPL(my_extension, CUDA, m) {
    m.impl("my_op", &my_op_cuda);
}
```

For Python-only operators, PyTorch 2.4+ introduces the streamlined `torch.library.custom_op()` API:

```python
@torch.library.custom_op("mylib::myop", mutates_args=())
def myop(x: torch.Tensor, y: float) -> torch.Tensor:
    # Implementation
    return result

@myop.register_fake
def _(x, y):
    # FakeTensor kernel for torch.compile
    return torch.empty_like(x)

# Autograd registration
myop.register_autograd(backward_fn, setup_context=setup_context_fn)
```

### CPU fallback implementation patterns

CPU implementations should use PyTorch's type-safe iteration patterns:

```cpp
at::Tensor my_op_cpu(const at::Tensor& input, const at::Tensor& other, double alpha) {
    TORCH_CHECK(input.device().is_cpu(), "input must be CPU tensor");
    
    auto result = torch::zeros_like(input);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_op_cpu", [&] {
        auto input_ptr = input.data_ptr<scalar_t>();
        auto other_ptr = other.data_ptr<scalar_t>();
        auto result_ptr = result.data_ptr<scalar_t>();
        
        for (int64_t i = 0; i < input.numel(); i++) {
            result_ptr[i] = input_ptr[i] * other_ptr[i] * alpha;
        }
    });
    
    return result;
}
```

### CUDA kernel implementation best practices

CUDA kernels should use PyTorch's `PackedTensorAccessor` for safe memory access:

```cuda
template <typename scalar_t>
__global__ void my_op_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> other,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> result,
    const scalar_t alpha) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input.size(0)) {
        result[idx] = input[idx] * other[idx] * alpha;
    }
}
```

## Distribution Strategies for Pre-compiled Binaries

### Wheel building for multiple CUDA versions

The most effective approach follows PyTorch's model of separate wheels for different CUDA versions:

```python
# Version naming with CUDA suffix
version = "1.0.0"
if torch.version.cuda:
    version += f"+cu{torch.version.cuda.replace('.', '')}"
```

**Key environment variables** for wheel building:
- `TORCH_CUDA_ARCH_LIST`: Specify target GPU architectures (e.g., "8.0 8.6+PTX")
- `CUDA_HOME`: CUDA installation path
- `MAX_JOBS`: Control parallel compilation jobs

### CI/CD strategies for multi-platform builds

Successful packages use GitHub Actions with matrix builds:

```yaml
strategy:
  matrix:
    python-version: [3.8, 3.9, 3.10, 3.11]
    cuda-version: [11.8, 12.1, 12.4]
    pytorch-version: [2.0, 2.1, 2.2]
    os: [ubuntu-latest, windows-latest]
```

**cibuildwheel integration** for CUDA extensions requires pre-installing CUDA in the build environment:

```yaml
- name: Build CUDA Linux Wheels
  run: python -m cibuildwheel --output-dir wheelhouse/cuda
  env:
    CIBW_BEFORE_ALL: >
      yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo &&
      yum clean all &&
      yum -y install cuda-toolkit
```

### Platform-specific considerations

**Linux** provides the best ecosystem support with `manylinux` wheels for broad compatibility. Use `auditwheel` for dependency bundling and consider `manylinux2014` or later for CUDA support.

**Windows** requires careful setup with Visual Studio Build Tools and MSVC compiler requirements. GitHub Actions with Windows runners provides the most reliable build environment.

**macOS** has limited CUDA support and should focus on CPU-only builds, potentially using Metal Performance Shaders for GPU acceleration.

## C API Exposure for LibTorch Users

### Operator registration for C++ consumption

The modern approach uses `TORCH_LIBRARY` macros that automatically expose operators to both Python and C++ environments:

```cpp
// C++ operator definition
TORCH_LIBRARY(custom_namespace, m) {
    m.def("myop(Tensor input, float param) -> Tensor");
}

// Implementation registration
TORCH_LIBRARY_IMPL(custom_namespace, CPU, m) {
    m.impl("myop", &myop_cpu);
}
```

### LibTorch integration patterns

For LibTorch-only deployments, operators can be loaded dynamically:

```cpp
// Load custom operator library
torch::jit::load("path/to/custom_ops.pt");

// Use operator in C++
torch::Tensor result = torch::ops::custom_namespace::myop(input, param);
```

### Memory management and tensor handling

LibTorch uses ATen's automatic memory management:

```cpp
// Safe tensor operations
torch::Tensor result = torch::empty_like(input);
const float* input_ptr = input.data_ptr<float>();
float* result_ptr = result.data_ptr<float>();

// ATen handles memory lifecycle automatically
```

## Analysis of Successful Implementation Examples

### NVIDIA Apex architectural patterns

Apex demonstrates **modular design** with separate components for different functionalities and **flexible installation** supporting both source builds and pre-compiled binaries. Their approach includes:

- **JIT compilation** for development and testing
- **Containerized distribution** through NGC containers
- **Graceful degradation** when CUDA extensions aren't available

### Flash Attention distribution strategies

Flash Attention succeeds through **hardware-specific optimization** with different implementations for different GPU generations and **PyPI distribution** of pre-compiled wheels for common configurations:

- **GPU architecture detection** during build
- **Multi-backend support** for CUDA, ROCm, and CPU
- **Version-specific optimization** for different hardware

### xFormers build system excellence

xFormers provides an excellent model for **multi-CUDA version support** and **component-based architecture**:

- **Pre-built wheels** available through PyPI with CUDA-specific versions
- **CUDA version matrix** supporting multiple CUDA versions (11.8, 12.6, 12.8)
- **Architecture targeting** using `TORCH_CUDA_ARCH_LIST`

## Current Best Practices and Recent Changes

### PyTorch 2.4+ API recommendations

The **new high-level APIs** provide better integration with modern PyTorch subsystems:

- **Use `torch.library.custom_op()`** for Python operators
- **Use `torch.library.triton_op()`** for Triton-based operators
- **Always use `torch.library.opcheck()`** for comprehensive testing

### Deprecated methods to avoid

PyTorch 2.4+ deprecates several older approaches:

- **Manual pybind11 bindings** for simple operators
- **Low-level torch.library APIs** for basic use cases
- **Direct torch.autograd.Function usage** due to composability issues

### Testing and validation requirements

Modern PyTorch requires comprehensive testing with `torch.library.opcheck()`:

```python
def test_with_opcheck():
    def sample_inputs():
        return [
            (torch.randn(5, 5), torch.randn(5, 5), 1.0),
            (torch.randn(10, 10), torch.randn(10, 10), 2.5),
        ]
    
    for args in sample_inputs():
        torch.library.opcheck(torch.ops.my_extension.my_op.default, args)
```

## Build System and Tooling Recommendations

### setuptools vs CMake decision framework

**Choose setuptools** for:
- Straightforward extensions primarily interacting with PyTorch
- Projects wanting native PyTorch integration
- Simple build requirements

**Choose CMake** for:
- Complex projects requiring extensive C++ development
- Existing CMake infrastructure
- Better IDE integration and debugging support

### Distribution tooling ecosystem

**cibuildwheel** excels for automated cross-platform wheel building but requires careful CUDA setup. **conda-forge** handles complex dependencies well and provides good CUDA support with automated maintenance.

**Custom solutions** work best for packages with unique requirements, as demonstrated by successful packages like torch-scatter and flash-attn.

## Conclusion

Creating PyTorch packages with custom CUDA kernels has become significantly more accessible with PyTorch 2.4's API overhaul. The new high-level APIs eliminate much of the complexity while providing better integration with modern PyTorch subsystems. Success depends on following current best practices: using modern registration APIs, implementing comprehensive testing, providing multiple distribution options, and learning from successful packages like Apex, Flash Attention, and xFormers.

The key to success lies in **starting with the modern PyTorch 2.4+ APIs**, implementing **robust build systems** that handle multiple CUDA versions, and providing **comprehensive testing** across target platforms. The ecosystem has matured significantly, making it easier than ever to create high-performance custom operators that integrate seamlessly with PyTorch's advanced features while maintaining broad compatibility and ease of use.