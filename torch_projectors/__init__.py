# torch_projectors/__init__.py

import torch
from pathlib import Path

# 1. Load the C++ extension library.
#    This must happen before the Python registration.
try:
    _lib_path = next((Path(__file__).parent).glob("_C*.so"))
except StopIteration:
    raise ImportError("Could not find C++ extension library. Did you compile the project?")
torch.ops.load_library(_lib_path)

# 2. Import the `ops` module to run the Python registration.
from . import ops

# 3. Expose the user-facing function.
from .ops import forward_project_2d, backward_project_2d

__all__ = ["forward_project_2d", "backward_project_2d"] 