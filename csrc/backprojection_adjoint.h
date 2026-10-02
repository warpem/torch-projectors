#pragma once

#include <torch/extension.h>

namespace torch_projectors {

// Backprojection inserts each retained input sample once, but an interpolation
// tap on kx=0 writes both it and its conjugate partner to the output. Reverse
// that duplication before gathering the output gradient. Self-conjugate taps
// were inserted only once and must not be doubled. This is deliberately local
// to backprojection backward: ordinary projection sampling has no such write.
// Use real views so this also works on MPS without complex arithmetic kernels.
inline at::Tensor fold_backprojection_boundary_gradient(const at::Tensor& grad) {
    const bool complex = grad.is_complex();
    const auto real_grad = complex ? at::view_as_real(grad.resolve_conj()) : grad;
    auto result = real_grad.clone(at::MemoryFormat::Contiguous);
    const int64_t column_axis = complex ? result.dim() - 2 : result.dim() - 1;
    const auto source_plane = real_grad.select(column_axis, 0);
    auto partner = source_plane.clone();
    // Plane dimensions are [batch, (depth), row, (real/imag)]. Negation of
    // FFT-order coordinates is reverse followed by a one-element roll.
    for (int64_t axis = 1; axis < column_axis; ++axis) {
        partner = at::roll(at::flip(partner, {axis}), {1}, {axis});
    }
    if (complex) partner.select(-1, 1).neg_();

    // The origin and Nyquist corners are their own partners. Forward's
    // insertion helper explicitly suppresses the second write at these taps.
    const int64_t plane_axes = column_axis - 1;
    for (int64_t corner = 0; corner < (int64_t(1) << plane_axes); ++corner) {
        auto self = partner;
        bool valid = true;
        for (int64_t axis = 1; axis <= plane_axes; ++axis) {
            const bool nyquist = (corner >> (axis - 1)) & 1;
            const int64_t size = source_plane.size(axis);
            if (nyquist && size % 2) { valid = false; break; }
            self = self.select(1, nyquist ? size / 2 : 0);
        }
        if (valid) self.zero_();
    }
    result.select(column_axis, 0).add_(partner);
    return complex ? at::view_as_complex(result) : result;
}

} // namespace torch_projectors
