from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch
import os
import platform
import subprocess
import sys
import pybind11

# Get PyTorch's library directory to find libomp
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')

# Base sources that are compiled on all platforms
sources = [
    "csrc/torch_projectors.cpp",
    "csrc/cpu/2d/projection_2d_kernels.cpp",
    "csrc/cpu/2d/backprojection_2d_kernels.cpp",
    "csrc/cpu/3d/projection_3d_to_2d_kernels.cpp",
]

# Platform-specific compilation flags
if platform.system() == "Windows":
    # MSVC-specific flags
    extra_compile_args = {"cxx": ["/O2", "/std:c++20"]}
    extra_link_args = []
    # OpenMP for MSVC
    extra_compile_args["cxx"].append("/openmp")
else:
    # GCC/Clang flags
    extra_compile_args = {"cxx": ["-O3", "-std=c++20"]}
    extra_link_args = []
    # Add OpenMP support (Unix platforms)
    # Try to detect available OpenMP library
    if platform.system() == "Darwin":
        # macOS: Check for LLVM OpenMP from PyTorch (uses .dylib extension)
        if os.path.exists(os.path.join(torch_lib_dir, "libomp.dylib")):
            extra_compile_args["cxx"].extend(["-Xpreprocessor", "-fopenmp"])
            extra_link_args.extend(["-L" + torch_lib_dir, "-lomp"])
        else:
            print("Warning: OpenMP not found on macOS, compiling without OpenMP support")
    elif os.path.exists("/usr/lib/x86_64-linux-gnu/libgomp.so") or os.path.exists("/usr/lib/aarch64-linux-gnu/libgomp.so") or os.path.exists("/usr/lib/libgomp.so"):
        # GNU OpenMP (most common on Linux)
        extra_compile_args["cxx"].extend(["-fopenmp"])
        extra_link_args.extend(["-lgomp"])
    elif os.path.exists(os.path.join(torch_lib_dir, "libomp.so")):
        # LLVM OpenMP from PyTorch (Linux)
        extra_compile_args["cxx"].extend(["-Xpreprocessor", "-fopenmp"])
        extra_link_args.extend(["-L" + torch_lib_dir, "-lomp"])
    else:
        # Fallback: try GNU OpenMP
        extra_compile_args["cxx"].extend(["-fopenmp"])
        extra_link_args.extend(["-lgomp"])

# Check for CUDA availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print(f"Platform: {platform.system()}")

cuda_available = torch.cuda.is_available()

# Set CUDA architectures if not specified and CUDA is available
if cuda_available and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"
    print(f"Set TORCH_CUDA_ARCH_LIST to: {os.environ['TORCH_CUDA_ARCH_LIST']}")
else:
    print(f"TORCH_CUDA_ARCH_LIST already set to: {os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not set')}")

use_cuda = cuda_available

# Add CUDA backend if available
if use_cuda:
    print("CUDA detected, enabling CUDA backend...")
    print(f"Adding CUDA source: csrc/cuda/2d/projection_2d_kernels.cu")
    sources.append("csrc/cuda/2d/projection_2d_kernels.cu")
    print(f"Adding CUDA source: csrc/cuda/3d/projection_3d_to_2d_kernels.cu")
    sources.append("csrc/cuda/3d/projection_3d_to_2d_kernels.cu")
    extra_compile_args["cxx"].append("-DUSE_CUDA")
    extra_compile_args["nvcc"] = ["-O3", "--use_fast_math", "-DUSE_CUDA"]
    print(f"CUDA compile args: {extra_compile_args['nvcc']}")
else:
    print("CUDA not available or disabled, using CPU backend only")

# Add MPS backend on macOS
if platform.system() == "Darwin":
    # Generate Metal shader headers before compilation
    print("Generating Metal shader headers...")
    script_path = os.path.join(os.path.dirname(__file__), "generate_metal_headers.py")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error generating Metal headers: {result.stderr}")
        sys.exit(1)
    print(result.stdout)
    
    sources.append("csrc/mps/2d/projection_2d_kernels.mm")
    sources.append("csrc/mps/2d/backprojection_2d_kernels.mm")
    sources.append("csrc/mps/3d/projection_3d_to_2d_kernels.mm")
    extra_compile_args["cxx"].extend(["-ObjC++", "-fobjc-arc", "-mmacosx-version-min=12.0"])
    extra_link_args.extend(["-framework", "Metal", "-framework", "MetalPerformanceShaders"])

# Choose the appropriate extension type based on CUDA availability
if use_cuda:
    extension_class = CUDAExtension
    print("Using CUDAExtension for compilation")
else:
    extension_class = CppExtension
    print("Using CppExtension for compilation")

print(f"Final sources list: {sources}")
print(f"Final compile args: {extra_compile_args}")
print(f"Final link args: {extra_link_args}")

setup(
    name="torch-projectors",
    version="0.1.0",
    author="[Your Name]",
    author_email="[Your Email]",
    description="Differentiable forward and backward projectors for cryo-EM.",
    ext_modules=[
        extension_class(
            name="torch_projectors._C",
            sources=sources,
            include_dirs=[pybind11.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["torch_projectors"],
) 