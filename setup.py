from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
import os

# Get PyTorch's library directory to find libomp
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')

setup(
    name="torch-projectors",
    version="0.1.0",
    author="[Your Name]",
    author_email="[Your Email]",
    description="Differentiable forward and backward projectors for cryo-EM.",
    ext_modules=[
        CppExtension(
            name="torch_projectors._C",
            sources=[
                "csrc/torch_projectors.cpp",
                "csrc/cpu/cpu_kernels.cpp",
            ],
            extra_compile_args={"cxx": ["-O3", "-std=c++20", "-Xpreprocessor", "-fopenmp"]},
            extra_link_args=["-L" + torch_lib_dir, "-lomp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["torch_projectors"],
) 