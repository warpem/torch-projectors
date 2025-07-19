# torch-projectors

A high-performance, differentiable 2D and 3D projection library for PyTorch, tailored for applications in cryogenic electron microscopy (cryo-EM), tomography, and related fields.

This project is currently under active development.

## Development Setup

These instructions outline how to set up a development environment for `torch-projectors`. We use `conda` to manage dependencies.

### 1. Create and Activate the Conda Environment

First, create a dedicated conda environment and install the required dependencies. All commands should be run from the root of this repository.

```bash
# Create the conda environment
conda create -n torch-projectors python=3.11 -y

# Activate the environment
conda activate torch-projectors

# Install PyTorch and pytest
conda install pytorch pytest -c pytorch -c conda-forge -y
```

### 2. Install the Project in Editable Mode

To build the C++ extension and install the package in a way that reflects code changes automatically, use the following command.

We use `pip` for the editable install, which handles the C++ extension compilation. Our project uses a `pyproject.toml` file to ensure that `setuptools` can find PyTorch during the build process.

```bash
# Install the package in editable mode
python -m pip install -e .
```

If the command succeeds, the C++ extension has been compiled successfully.

Our project uses the modern PyTorch C++ extension hybrid approach:
*   **C++ Kernels:** The performance-sensitive forward and backward passes are written as separate C++ functions.
*   **`TORCH_LIBRARY` Registration:** Both the forward and backward C++ kernels are registered as distinct PyTorch operators.
*   **Python `autograd` Registration:** In Python, `torch.library.register_autograd` is used to link the two C++ operators. It tells the autograd engine that when it sees our forward operator, it should use our C++ backward operator in the backward pass. This pattern provides a stable, high-performance, and officially recommended way to create differentiable custom operators.

### 3. Running Tests

Tests are managed using `pytest`. To run the test suite, execute the following command from the repository root:

```bash
pytest
```