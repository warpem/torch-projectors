/**
 * Projection Utilities and Common Functions
 * 
 * This header provides shared utilities for projection operations including
 * input validation, tensor setup, parallel processing utilities, and 
 * phase shift computations.
 */

#pragma once

#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <c10/util/Optional.h>
#include <cmath>

namespace torch_projectors {
namespace cpu {
namespace common {

/**
 * Validate common projection parameters
 */
inline void validate_projection_inputs(
    const std::string& interpolation,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    int expected_rec_dims,
    int expected_rotation_size
) {
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic", 
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(reconstruction.is_complex(), "Reconstruction must be a complex tensor");
    TORCH_CHECK(reconstruction.dim() == expected_rec_dims, 
                "Reconstruction must have " + std::to_string(expected_rec_dims) + " dimensions");
    TORCH_CHECK(rotations.dim() == 4 && 
                rotations.size(2) == expected_rotation_size && 
                rotations.size(3) == expected_rotation_size, 
                "Rotations must be (B_rot, P, " + std::to_string(expected_rotation_size) + 
                ", " + std::to_string(expected_rotation_size) + ")");
    
    if (shifts.has_value()) {
        TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
        TORCH_CHECK(shifts->size(2) == 2, "Shifts must have 2 components (x, y)");
        TORCH_CHECK(shifts->scalar_type() == rotations.scalar_type(), 
                    "Shifts and rotations must have the same dtype");
    }
}

/**
 * Compute optimal grain size for parallel processing
 */
inline int64_t compute_grain_size(int64_t total_work, int64_t min_grain_size = 1) {
    return std::max(min_grain_size, total_work / (2 * at::get_num_threads()));
}

/**
 * Convert array indices to Fourier coordinates
 * Handles FFTW format: DC at (0,0), positive frequencies, then negative
 */
template <typename real_t>
inline real_t index_to_fourier_coord(int64_t index, int64_t boxsize) {
    return (index <= boxsize / 2) ? static_cast<real_t>(index) : static_cast<real_t>(index - boxsize);
}

/**
 * Check if frequency should be filtered out based on radius cutoff
 */
template <typename real_t>
inline bool should_filter_frequency(const std::vector<real_t>& coords, real_t radius_cutoff_sq) {
    real_t radius_sq = 0;
    for (const auto& coord : coords) {
        radius_sq += coord * coord;
    }
    return radius_sq > radius_cutoff_sq;
}

/**
 * Compute phase factor for translation shifts
 * Phase = -2π * (k · shift) / boxsize
 */
template <typename scalar_t, typename coord_real_t, typename shift_real_t>
inline scalar_t compute_phase_factor(
    const std::vector<coord_real_t>& fourier_coords,
    const std::vector<shift_real_t>& shifts,
    coord_real_t boxsize
) {
    coord_real_t phase = 0.0;
    for (size_t i = 0; i < fourier_coords.size() && i < shifts.size(); ++i) {
        phase += fourier_coords[i] * static_cast<coord_real_t>(shifts[i]);
    }
    phase *= -2.0 * M_PI / boxsize;
    return scalar_t(cos(phase), sin(phase));
}

/**
 * Setup gradient tensors based on requires_grad flags
 */
template <typename TensorType>
struct GradientTensors {
    TensorType grad_reconstruction;
    TensorType grad_rotations;
    TensorType grad_shifts;
    bool need_rotation_grads;
    bool need_shift_grads;
    
    GradientTensors(
        const TensorType& grad_projections,
        const TensorType& reconstruction,
        const TensorType& rotations,
        const c10::optional<TensorType>& shifts
    ) : need_rotation_grads(rotations.requires_grad()),
        need_shift_grads(shifts.has_value() && shifts->requires_grad()) {
        
        // Always compute reconstruction gradients
        auto rec_sizes = reconstruction.sizes().vec();
        grad_reconstruction = torch::zeros(rec_sizes, grad_projections.options());
        
        // Conditional gradient tensor creation
        if (need_rotation_grads) {
            grad_rotations = torch::zeros_like(rotations);
        } else {
            grad_rotations = torch::empty({0}, rotations.options());
        }
        
        if (need_shift_grads) {
            grad_shifts = torch::zeros_like(*shifts);
        } else {
            grad_shifts = torch::empty({0}, grad_projections.options().dtype(rotations.scalar_type()));
        }
    }
};

/**
 * Batch indexing helper for broadcasting support
 */
struct BatchIndices {
    int64_t rot_b_idx;
    int64_t shift_b_idx;
    
    BatchIndices(int64_t b, int64_t B_rot, int64_t B_shift) 
        : rot_b_idx((B_rot == 1) ? 0 : b),
          shift_b_idx((B_shift == 1) ? 0 : b) {}
};

// Forward projection coordinate helpers are implemented directly in the projection files
// to avoid template complexity issues

/**
 * Backward projection kernel interface
 * Provides uniform interface for gradient distribution
 */
template <int N, typename scalar_t, typename real_t = typename scalar_t::value_type>
struct BackwardKernel {
    virtual ~BackwardKernel() = default;
    
    /**
     * Distribute gradient from a single point to reconstruction neighborhood
     */
    virtual void distribute_gradient(
        std::function<void(const std::array<int64_t, N>&, scalar_t)> accumulate_func,
        scalar_t grad_val,
        const std::array<real_t, N>& coords
    ) const = 0;
};

/**
 * Linear backward kernel for 2D
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct Linear2DBackwardKernel : public BackwardKernel<2, scalar_t, real_t> {
    void distribute_gradient(
        std::function<void(const std::array<int64_t, 2>&, scalar_t)> accumulate_func,
        scalar_t grad_val,
        const std::array<real_t, 2>& coords
    ) const override {
        // Get floor and fractional parts
        int64_t r_floor = floor(coords[0]);
        int64_t c_floor = floor(coords[1]);
        real_t r_frac = coords[0] - r_floor;
        real_t c_frac = coords[1] - c_floor;
        
        // Distribute to 4 corners of 2D square
        accumulate_func({r_floor,     c_floor},     grad_val * (1 - r_frac) * (1 - c_frac));
        accumulate_func({r_floor + 1, c_floor},     grad_val * r_frac * (1 - c_frac));
        accumulate_func({r_floor,     c_floor + 1}, grad_val * (1 - r_frac) * c_frac);
        accumulate_func({r_floor + 1, c_floor + 1}, grad_val * r_frac * c_frac);
    }
};

/**
 * Linear backward kernel for 3D
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct Linear3DBackwardKernel : public BackwardKernel<3, scalar_t, real_t> {
    void distribute_gradient(
        std::function<void(const std::array<int64_t, 3>&, scalar_t)> accumulate_func,
        scalar_t grad_val,
        const std::array<real_t, 3>& coords
    ) const override {
        // Get floor and fractional parts
        int64_t d_floor = floor(coords[0]);
        int64_t r_floor = floor(coords[1]);
        int64_t c_floor = floor(coords[2]);
        real_t d_frac = coords[0] - d_floor;
        real_t r_frac = coords[1] - r_floor;
        real_t c_frac = coords[2] - c_floor;
        
        // Distribute to 8 corners of 3D cube
        accumulate_func({d_floor,     r_floor,     c_floor},     grad_val * (1 - d_frac) * (1 - r_frac) * (1 - c_frac));
        accumulate_func({d_floor + 1, r_floor,     c_floor},     grad_val * d_frac * (1 - r_frac) * (1 - c_frac));
        accumulate_func({d_floor,     r_floor + 1, c_floor},     grad_val * (1 - d_frac) * r_frac * (1 - c_frac));
        accumulate_func({d_floor + 1, r_floor + 1, c_floor},     grad_val * d_frac * r_frac * (1 - c_frac));
        accumulate_func({d_floor,     r_floor,     c_floor + 1}, grad_val * (1 - d_frac) * (1 - r_frac) * c_frac);
        accumulate_func({d_floor + 1, r_floor,     c_floor + 1}, grad_val * d_frac * (1 - r_frac) * c_frac);
        accumulate_func({d_floor,     r_floor + 1, c_floor + 1}, grad_val * (1 - d_frac) * r_frac * c_frac);
        accumulate_func({d_floor + 1, r_floor + 1, c_floor + 1}, grad_val * d_frac * r_frac * c_frac);
    }
};

/**
 * Cubic backward kernel for 2D
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct Cubic2DBackwardKernel : public BackwardKernel<2, scalar_t, real_t> {
    void distribute_gradient(
        std::function<void(const std::array<int64_t, 2>&, scalar_t)> accumulate_func,
        scalar_t grad_val,
        const std::array<real_t, 2>& coords
    ) const override {
        // Get floor and fractional parts
        int64_t r_floor = floor(coords[0]);
        int64_t c_floor = floor(coords[1]);
        real_t r_frac = coords[0] - r_floor;
        real_t c_frac = coords[1] - c_floor;
        
        // Distribute to 4x4 grid around the point (cubic support)
        for (int r_offset = -1; r_offset <= 2; ++r_offset) {
            real_t r_weight = cubic_kernel(r_frac - r_offset);
            if (r_weight == 0.0) continue;
            
            for (int c_offset = -1; c_offset <= 2; ++c_offset) {
                real_t c_weight = cubic_kernel(c_frac - c_offset);
                if (c_weight == 0.0) continue;
                
                accumulate_func({r_floor + r_offset, c_floor + c_offset}, 
                              grad_val * r_weight * c_weight);
            }
        }
    }

private:
    real_t cubic_kernel(real_t s) const {
        return torch_projectors::cpu::common::cubic_kernel(s);
    }
};

/**
 * Cubic backward kernel for 3D
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct Cubic3DBackwardKernel : public BackwardKernel<3, scalar_t, real_t> {
    void distribute_gradient(
        std::function<void(const std::array<int64_t, 3>&, scalar_t)> accumulate_func,
        scalar_t grad_val,
        const std::array<real_t, 3>& coords
    ) const override {
        // Get floor and fractional parts
        int64_t d_floor = floor(coords[0]);
        int64_t r_floor = floor(coords[1]);
        int64_t c_floor = floor(coords[2]);
        real_t d_frac = coords[0] - d_floor;
        real_t r_frac = coords[1] - r_floor;
        real_t c_frac = coords[2] - c_floor;
        
        // Distribute to 4x4x4 grid around the point (cubic support)
        for (int d_offset = -1; d_offset <= 2; ++d_offset) {
            real_t d_weight = cubic_kernel(d_frac - d_offset);
            if (d_weight == 0.0) continue;
            
            for (int r_offset = -1; r_offset <= 2; ++r_offset) {
                real_t r_weight = cubic_kernel(r_frac - r_offset);
                if (r_weight == 0.0) continue;
                
                for (int c_offset = -1; c_offset <= 2; ++c_offset) {
                    real_t c_weight = cubic_kernel(c_frac - c_offset);
                    if (c_weight == 0.0) continue;
                    
                    accumulate_func({d_floor + d_offset, r_floor + r_offset, c_floor + c_offset}, 
                                  grad_val * d_weight * r_weight * c_weight);
                }
            }
        }
    }

private:
    real_t cubic_kernel(real_t s) const {
        return torch_projectors::cpu::common::cubic_kernel(s);
    }
};

/**
 * Factory for 2D backward kernels
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
std::unique_ptr<BackwardKernel<2, scalar_t, real_t>> get_2d_backward_kernel(const std::string& interpolation) {
    if (interpolation == "linear") {
        return std::make_unique<Linear2DBackwardKernel<scalar_t, real_t>>();
    } else if (interpolation == "cubic") {
        return std::make_unique<Cubic2DBackwardKernel<scalar_t, real_t>>();
    } else {
        throw std::runtime_error("Unsupported 2D backward interpolation method: " + interpolation);
    }
}

/**
 * Factory for 3D backward kernels
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
std::unique_ptr<BackwardKernel<3, scalar_t, real_t>> get_3d_backward_kernel(const std::string& interpolation) {
    if (interpolation == "linear") {
        return std::make_unique<Linear3DBackwardKernel<scalar_t, real_t>>();
    } else if (interpolation == "cubic") {
        return std::make_unique<Cubic3DBackwardKernel<scalar_t, real_t>>();
    } else {
        throw std::runtime_error("Unsupported 3D backward interpolation method: " + interpolation);
    }
}

/**
 * 2D Gradient accumulator with Friedel symmetry handling
 * 
 * Handles atomic accumulation of gradients into 2D reconstruction with proper
 * Friedel symmetry and bounds checking.
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline void accumulate_2d_gradient(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& grad_rec_acc,
    int64_t b, int64_t rec_boxsize, int64_t rec_boxsize_half,
    int64_t r, int64_t c, scalar_t grad
) {
    bool needs_conj = false;
    
    // Handle Friedel symmetry for negative column indices
    if (c < 0) { 
        c = -c;
        r = -r;
        needs_conj = true;
    }
    
    // Bounds checking
    if (c >= rec_boxsize_half) return;
    if (r > rec_boxsize / 2 || r < -rec_boxsize / 2 + 1) return;

    // Convert negative row indices to positive (FFTW wrapping)
    int64_t r_eff = r < 0 ? rec_boxsize + r : r;
    if (r_eff >= rec_boxsize) return;

    // Atomically accumulate gradient (with conjugation if needed for Friedel symmetry)
    scalar_t final_grad = needs_conj ? std::conj(grad) : grad;
    atomic_add_complex(&grad_rec_acc[b][r_eff][c], final_grad);
}


/**
 * 2D Data accumulator with Friedel symmetry handling
 * 
 * Functions like accumulate_2d_gradient but makes sure 
 * components at c = 0 are inserted twice (Friedel-symmetrically)
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline void accumulate_2d_data(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& grad_rec_acc,
    int64_t b, int64_t rec_boxsize, int64_t rec_boxsize_half,
    int64_t r, int64_t c, scalar_t grad
) {
    bool needs_conj = false;
    
    // Handle Friedel symmetry for negative column indices
    if (c < 0) { 
        c = -c;
        r = -r;
        needs_conj = true;
    }
    
    // Bounds checking
    if (c >= rec_boxsize_half) return;
    if (r > rec_boxsize / 2 || r < -rec_boxsize / 2 + 1) return;

    // Convert negative row indices to positive (FFTW wrapping)
    int64_t r_eff = r < 0 ? rec_boxsize + r : r;
    if (r_eff >= rec_boxsize) return;

    // Atomically accumulate gradient (with conjugation if needed for Friedel symmetry)
    scalar_t final_grad = needs_conj ? std::conj(grad) : grad;
    atomic_add_complex(&grad_rec_acc[b][r_eff][c], final_grad);

    // On the x=0 line, also insert Friedel-symmetric conjugate counterpart
    if (c == 0) {
        r *= -1;
        int64_t r_eff2 = r < 0 ? rec_boxsize + r : r;
        if (r_eff2 >= rec_boxsize || r_eff2 == r_eff) return;

        atomic_add_complex(&grad_rec_acc[b][r_eff2][c], std::conj(final_grad));
    }
}

/**
 * 3D Gradient accumulator with Friedel symmetry handling
 * 
 * Handles atomic accumulation of gradients into 3D reconstruction with proper
 * 3D Friedel symmetry and bounds checking.
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline void accumulate_3d_gradient(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& grad_rec_acc,
    int64_t b, int64_t boxsize, int64_t rec_boxsize_half,
    int64_t d, int64_t r, int64_t c, scalar_t grad
) {
    bool needs_conj = false;
    
    // Handle 3D Friedel symmetry for negative column indices
    if (c < 0) { 
        c = -c;
        r = -r;
        d = -d;
        needs_conj = true;
    }
    
    // Bounds checking
    if (c >= rec_boxsize_half) return;
    if (r > boxsize / 2 || r < -boxsize / 2 + 1) return;
    if (d > boxsize / 2 || d < -boxsize / 2 + 1) return;

    // Convert negative indices to positive (FFTW wrapping)
    int64_t r_eff = r < 0 ? boxsize + r : r;
    int64_t d_eff = d < 0 ? boxsize + d : d;
    if (r_eff >= boxsize || d_eff >= boxsize) return;

    // Accumulate gradient with conjugation for Friedel symmetry
    scalar_t final_grad = needs_conj ? std::conj(grad) : grad;
    atomic_add_complex(&grad_rec_acc[b][d_eff][r_eff][c], final_grad);
}

/**
 * 3D Gradient accumulator with Friedel symmetry handling
 * 
 * Handles atomic accumulation of gradients into 3D reconstruction with proper
 * 3D Friedel symmetry and bounds checking.
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline void accumulate_3d_data(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& grad_rec_acc,
    int64_t b, int64_t boxsize, int64_t rec_boxsize_half,
    int64_t d, int64_t r, int64_t c, scalar_t grad
) {
    bool needs_conj = false;
    
    // Handle 3D Friedel symmetry for negative column indices
    if (c < 0) { 
        c = -c;
        r = -r;
        d = -d;
        needs_conj = true;
    }
    
    // Bounds checking
    if (c >= rec_boxsize_half) return;
    if (r > boxsize / 2 || r < -boxsize / 2 + 1) return;
    if (d > boxsize / 2 || d < -boxsize / 2 + 1) return;

    // Convert negative indices to positive (FFTW wrapping)
    int64_t r_eff = r < 0 ? boxsize + r : r;
    int64_t d_eff = d < 0 ? boxsize + d : d;
    if (r_eff >= boxsize || d_eff >= boxsize) return;

    // Accumulate gradient with conjugation for Friedel symmetry
    scalar_t final_grad = needs_conj ? std::conj(grad) : grad;
    atomic_add_complex(&grad_rec_acc[b][d_eff][r_eff][c], final_grad);

    // On the x=0 line, also insert Friedel-symmetric conjugate counterpart
    if (c == 0) {
        r *= -1;
        d *= -1;
        int64_t d_eff2 = d < 0 ? boxsize + d : d;
        int64_t r_eff2 = r < 0 ? boxsize + r : r;
        if (d_eff2 >= boxsize || 
            r_eff2 >= boxsize || 
            (r_eff2 == r_eff && d_eff2 == d_eff)) return;

        atomic_add_complex(&grad_rec_acc[b][d_eff2][r_eff2][c], std::conj(final_grad));
    }
}

} // namespace common
} // namespace cpu
} // namespace torch_projectors