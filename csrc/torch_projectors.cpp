#include <torch/extension.h>
#include "cpu/cpu_kernels.h"

// Register our C++ functions as PyTorch operators.
// We register both the forward and backward functions as separate operators.
TORCH_LIBRARY(torch_projectors, m) {
  m.def("add_tensors_forward(Tensor a, Tensor b) -> Tensor");
  m.def("add_tensors_backward(Tensor grad_output) -> (Tensor, Tensor)");
}

// Register the C++ implementations for the CPU backend for both operators.
TORCH_LIBRARY_IMPL(torch_projectors, CPU, m) {
  m.impl("add_tensors_forward", &add_tensors_forward_cpu);
  m.impl("add_tensors_backward", &add_tensors_backward_cpu);
} 