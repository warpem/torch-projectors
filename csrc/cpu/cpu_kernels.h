#pragma once
#include <torch/extension.h>

at::Tensor forward_project_2d_cpu(
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef output_shape,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

at::Tensor backward_project_2d_cpu(
    const at::Tensor& projections,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef reconstruction_shape,
    const std::string& interpolation,
    const double oversampling
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_project_2d_cpu_adj(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
); 