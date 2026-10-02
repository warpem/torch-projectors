#pragma once

#include "interpolation_kernels.h"

namespace torch_projectors { namespace cpu { namespace common {

// Unlike projection's edge-clamped sampler, this gather is the transpose of
// backprojection insertion: taps outside the Fourier box contribute zero.
// The kx=0 symmetry-plane duplication has already been folded into rec.
template <int N, typename scalar_t, typename real_t = typename scalar_t::value_type>
class BackprojectionSamplingKernel : public InterpolationKernel<N, scalar_t, real_t> {
    bool cubic_;
    using Accessor = torch::PackedTensorAccessor32<scalar_t, N + 1, torch::DefaultPtrTraits>;

    scalar_t sample(const Accessor& rec, int64_t b, int64_t size, int64_t half,
                    std::array<int64_t, N> point) const {
        const bool conjugate = point[N - 1] < 0;
        if (conjugate) for (auto& x : point) x = -x;
        if (point[N - 1] >= half) return scalar_t(0, 0);
        for (int axis = 0; axis < N - 1; ++axis) {
            if (point[axis] > size / 2 || point[axis] < -size / 2 + 1)
                return scalar_t(0, 0);
            if (point[axis] < 0) point[axis] += size;
        }
        scalar_t value;
        if constexpr (N == 2) value = rec[b][point[0]][point[1]];
        else value = rec[b][point[0]][point[1]][point[2]];
        return conjugate ? std::conj(value) : value;
    }

public:
    explicit BackprojectionSamplingKernel(const std::string& interpolation)
        : cubic_(interpolation == "cubic") {}

    scalar_t interpolate(const Accessor& rec, const int64_t b, const int64_t size,
                         const int64_t half, const std::array<real_t, N>& coords) const override {
        return std::get<0>(interpolate_with_gradients(rec, b, size, half, coords));
    }

    std::tuple<scalar_t, std::array<scalar_t, N>> interpolate_with_gradients(
        const Accessor& rec, const int64_t b, const int64_t size,
        const int64_t half, const std::array<real_t, N>& coords) const override {
        const int width = cubic_ ? 4 : 2;
        std::array<int64_t, N> base;
        real_t weights[N][4], derivatives[N][4];
        int taps = 1;
        for (int axis = 0; axis < N; ++axis) {
            const int64_t floor_coord = std::floor(coords[axis]);
            base[axis] = floor_coord - (cubic_ ? 1 : 0);
            const real_t fraction = coords[axis] - floor_coord;
            for (int tap = 0; tap < width; ++tap) {
                const real_t delta = coords[axis] - (base[axis] + tap);
                weights[axis][tap] = cubic_ ? cubic_kernel(delta) : (tap ? fraction : 1 - fraction);
                derivatives[axis][tap] = cubic_ ? cubic_kernel_derivative(delta) : (tap ? 1 : -1);
            }
            taps *= width;
        }
        scalar_t value(0, 0);
        std::array<scalar_t, N> gradient{};
        for (int flat = 0; flat < taps; ++flat) {
            int remaining = flat;
            std::array<int, N> tap;
            std::array<int64_t, N> point;
            real_t weight = 1;
            for (int axis = 0; axis < N; ++axis) {
                tap[axis] = remaining % width; remaining /= width;
                point[axis] = base[axis] + tap[axis];
                weight *= weights[axis][tap[axis]];
            }
            const scalar_t sampled = sample(rec, b, size, half, point);
            value += sampled * weight;
            for (int derivative_axis = 0; derivative_axis < N; ++derivative_axis) {
                real_t derivative = 1;
                for (int axis = 0; axis < N; ++axis)
                    derivative *= axis == derivative_axis ? derivatives[axis][tap[axis]] : weights[axis][tap[axis]];
                gradient[derivative_axis] += sampled * derivative;
            }
        }
        return std::make_tuple(value, gradient);
    }
};

}}} // namespace torch_projectors::cpu::common
