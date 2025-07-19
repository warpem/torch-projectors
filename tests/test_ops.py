import torch
import torch_projectors
import pytest
from torch.autograd import gradcheck

def test_add_tensors_forward():
    """
    Tests the forward pass of the custom add_tensors operator.
    """
    a = torch.randn(5, 5, dtype=torch.double)
    b = torch.randn(5, 5, dtype=torch.double)
    c = torch_projectors.add_tensors(a, b)
    assert torch.allclose(c, a + b)

def test_add_tensors_backward():
    """
    Tests the backward pass of the custom add_tensors operator using gradcheck.
    """
    a = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
    b = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
    test = gradcheck(torch_projectors.add_tensors, (a, b), eps=1e-6, atol=1e-4)
    assert test, "Gradient check failed for add_tensors"

def test_add_tensors_error():
    """
    Tests that the operator raises an error for mismatched shapes.
    """
    a = torch.randn(5, 5)
    b = torch.randn(3, 3)
    with pytest.raises(RuntimeError, match="Input tensors must have the same shape"):
        torch_projectors.add_tensors(a, b) 