#include "cpu_kernels.h"
#include <torch/extension.h>

at::Tensor add_tensors_forward_cpu(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu(), "Input tensors must be on the CPU");

    auto result = torch::empty_like(a);
    
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(a.scalar_type(), "add_tensors_forward_cpu", [&] {
        const auto* a_ptr = a.data_ptr<scalar_t>();
        const auto* b_ptr = b.data_ptr<scalar_t>();
        auto* result_ptr = result.data_ptr<scalar_t>();
        for (int64_t i = 0; i < a.numel(); ++i) {
            result_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    });
    
    return result;
}

std::tuple<at::Tensor, at::Tensor> add_tensors_backward_cpu(const at::Tensor& grad_output) {
    // The backward of addition is just passing the gradient through.
    // A manual implementation isn't very interesting here, but for more
    // complex operators, this function would contain significant logic.
    return {grad_output.clone(), grad_output.clone()};
} 