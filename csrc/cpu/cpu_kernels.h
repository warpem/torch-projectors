#pragma once
#include <torch/extension.h>

// Explicit C++ implementation for the forward pass
at::Tensor add_tensors_forward_cpu(const at::Tensor& a, const at::Tensor& b);

// Explicit C++ implementation for the backward pass
std::tuple<at::Tensor, at::Tensor> add_tensors_backward_cpu(const at::Tensor& grad_output); 