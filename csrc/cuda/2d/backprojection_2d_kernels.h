#pragma once
#include <torch/extension.h>

#ifdef USE_CUDA

std::tuple<at::Tensor, at::Tensor> backproject_2d_forw_cuda(
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backproject_2d_back_cuda(
    const at::Tensor& grad_data_rec,
    const c10::optional<at::Tensor>& grad_weight_rec,
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
);

#endif // USE_CUDA