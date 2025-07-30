#pragma once
#include <torch/extension.h>

#ifdef USE_CUDA

at::Tensor forward_project_2d_cuda(
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef output_shape,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_project_2d_cuda(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

#endif // USE_CUDA