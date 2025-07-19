# torch_projectors/ops.py

import torch

# This is the user-facing Python function.
def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Adds two tensors using a custom C++ kernel for the forward pass.
    The backward pass is also a C++ kernel, linked via the Python registration below.
    """
    # This calls our C++ FORWARD kernel.
    return torch.ops.torch_projectors.add_tensors_forward(a, b)

# --- Modern Autograd Registration ---

def _add_tensors_backward(ctx, grad_output):
    """
    This Python function is the backward formula.
    It simply calls our C++ BACKWARD kernel.
    """
    grad_a, grad_b = torch.ops.torch_projectors.add_tensors_backward(grad_output)
    return grad_a, grad_b

# Link the backward formula to the FORWARD operator.
torch.library.register_autograd(
    "torch_projectors::add_tensors_forward",
    _add_tensors_backward
) 