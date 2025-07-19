from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

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
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["torch_projectors"],
) 