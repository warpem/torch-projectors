from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch
import os
import platform
import subprocess
import sys

# Get PyTorch's library directory to find libomp
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')

# Base sources that are compiled on all platforms
sources = [
    "csrc/torch_projectors.cpp",
    "csrc/cpu/cpu_kernels.cpp",
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
    extra_compile_args["cxx"].extend(["-Xpreprocessor", "-fopenmp"])
    extra_link_args.extend(["-L" + torch_lib_dir, "-lomp"])

# Check for CUDA availability
cuda_available = torch.cuda.is_available()
use_cuda = cuda_available and os.environ.get("TORCH_CUDA_ARCH_LIST") is not None

# Add CUDA backend if available and requested
if use_cuda:
    print("CUDA detected, enabling CUDA backend...")
    sources.append("csrc/cuda/cuda_kernels.cu")
    extra_compile_args["cxx"].append("-DUSE_CUDA")
    extra_compile_args["nvcc"] = ["-O3", "--use_fast_math", "-DUSE_CUDA"]
    
    # Set CUDA architectures if not specified
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"

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
    
    sources.append("csrc/mps/mps_kernels.mm")
    extra_compile_args["cxx"].extend(["-ObjC++", "-fobjc-arc", "-mmacosx-version-min=12.0"])
    extra_link_args.extend(["-framework", "Metal", "-framework", "MetalPerformanceShaders"])

# Choose the appropriate extension type based on CUDA availability
if use_cuda:
    extension_class = CUDAExtension
else:
    extension_class = CppExtension

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
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["torch_projectors"],
) 