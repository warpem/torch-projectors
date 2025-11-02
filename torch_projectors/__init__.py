# torch_projectors/__init__.py

import torch
from pathlib import Path

# 1. Load the C++ extension library.
#    This must happen before the Python registration.
try:
    # Try platform-specific extensions: .pyd (Windows), .so (Linux), .dylib (macOS)
    _lib_path = next((Path(__file__).parent).glob("_C*.*"), None)
    if _lib_path is None or _lib_path.suffix not in ['.so', '.pyd', '.dylib']:
        raise StopIteration
except StopIteration:
    raise ImportError("Could not find C++ extension library. Did you compile the project?")
torch.ops.load_library(_lib_path)

# 2. Import the `ops` module to run the Python registration.
from . import ops

# 3. Expose the user-facing function.
from .ops import (
    project_2d_forw,
    project_2d_back,
    project_3d_to_2d_forw,
    project_3d_to_2d_back,
    backproject_2d_forw,
    backproject_2d_to_3d_forw,
    backproject_2d_to_3d_back,
    project_3d_forw,
    project_3d_back,
)

__all__ = [
    "project_2d_forw",
    "project_2d_back",
    "project_3d_to_2d_forw",
    "project_3d_to_2d_back",
    "backproject_2d_forw",
    "backproject_2d_to_3d_forw",
    "backproject_2d_to_3d_back",
    "project_3d_forw",
    "project_3d_back"
]
