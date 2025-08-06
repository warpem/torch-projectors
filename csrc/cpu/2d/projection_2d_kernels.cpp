/**
 * CPU Kernels for Differentiable 2D Projection Operations in Fourier Space
 * 
 * Refactored to use shared component headers for better code reuse and maintainability.
 * This file now focuses on 2D-specific projection logic while reusing common components.
 * 
 * Key Features:
 * - Forward projection: Sample from 3D Fourier reconstruction to create 2D projections
 * - Backward projection: Scatter 2D projection data into 3D Fourier reconstruction
 * - Uses shared interpolation kernels, FFTW sampling, and atomic operations
 * - Follows the Projection-Slice Theorem in Fourier space
 */

#include "projection_2d_kernels.h"
#include "../common/atomic_ops.h"
#include "../common/cubic_kernels.h"
#include "../common/fftw_sampling.h" 
#include "../common/interpolation_kernels.h"
#include "../common/projection_utils.h"
#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <complex>
#include <algorithm>
#include <omp.h>

using namespace torch_projectors::cpu::common;

/**
 * 2D Backward projection kernel for gradient distribution
 * 
 * Wraps the common backward kernel interface to work with the legacy API
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
class BackwardProjection2DKernel {
private:
    std::unique_ptr<BackwardKernel<2, scalar_t, real_t>> kernel_;

public:
    BackwardProjection2DKernel(const std::string& interpolation) 
        : kernel_(get_2d_backward_kernel<scalar_t, real_t>(interpolation)) {}
    
    void distribute_gradient(
        std::function<void(int64_t, int64_t, scalar_t)> accumulate_func,
        scalar_t grad_val,
        real_t r, real_t c
    ) const {
        // Convert legacy API to new array-based API
        auto new_accumulate_func = [&accumulate_func](const std::array<int64_t, 2>& indices, scalar_t grad) {
            accumulate_func(indices[0], indices[1], grad);
        };
        
        std::array<real_t, 2> coords = {r, c};
        kernel_->distribute_gradient(new_accumulate_func, grad_val, coords);
    }
};

/**
 * 2D Interpolation kernel wrapper
 * 
 * Wraps the common interpolation interface to work with the legacy API
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
class Interpolation2DKernel {
private:
    std::unique_ptr<InterpolationKernel<2, scalar_t, real_t>> kernel_;

public:
    Interpolation2DKernel(const std::string& interpolation) 
        : kernel_(get_interpolation_kernel<2, scalar_t, real_t>(interpolation)) {}
    
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const {
        std::array<real_t, 2> coords = {r, c};
        return kernel_->interpolate(rec, b, boxsize, boxsize_half, coords);
    }
    
    std::tuple<scalar_t, scalar_t, scalar_t> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const {
        std::array<real_t, 2> coords = {r, c};
        auto [val, grads] = kernel_->interpolate_with_gradients(rec, b, boxsize, boxsize_half, coords);
        return std::make_tuple(val, grads[0], grads[1]);
    }
};

/**
 * Forward projection from 3D Fourier reconstruction to 2D projections
 * 
 * Streamlined version using shared components for validation, setup, and interpolation.
 */
at::Tensor project_2d_forw_cpu(
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef output_shape,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Use shared validation
    validate_projection_inputs(interpolation, reconstruction, rotations, shifts, 3, 2);

    // Extract tensor dimensions
    const auto B = reconstruction.size(0);
    const auto boxsize = reconstruction.size(1);
    const auto boxsize_half = reconstruction.size(2);
    
    const auto B_rot = rotations.size(0);
    const auto P = rotations.size(1);
    TORCH_CHECK(B_rot == B || B_rot == 1, "Batch size of rotations must be 1 or same as reconstruction");

    const auto proj_boxsize = output_shape[0];
    const auto proj_boxsize_half = output_shape[0] / 2 + 1;
    
    // Initialize output tensor
    auto projection = torch::zeros({B, P, proj_boxsize, proj_boxsize_half}, reconstruction.options());

    // PyTorch dispatch for type-generic code
    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "forward_project_2d_cpu_rotations", ([&] {
        using rot_real_t = scalar_t;
        auto rot_acc = rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        
        // Handle optional shifts
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> shifts_acc;
        if (shifts.has_value()) {
            TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
            TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1, "Batch size of shifts must be 1 or same as reconstruction");
            TORCH_CHECK(shifts->size(1) == P, "Number of poses in shifts must match rotations");
            TORCH_CHECK(shifts->scalar_type() == rotations.scalar_type(), "Shifts and rotations must have the same dtype");
            shifts_acc.emplace(shifts->packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
        }
        
        // Dispatch for complex types
        AT_DISPATCH_COMPLEX_TYPES(reconstruction.scalar_type(), "forward_project_2d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;
            auto rec_acc = reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();
            auto proj_acc = projection.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();

            // Set up Fourier space filtering
            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;

            // Create interpolation kernel using shared components
            Interpolation2DKernel<scalar_t, real_t> kernel(interpolation);

            // Main projection loop with shared grain size computation
            const int64_t grain_size = compute_grain_size(B * P);
            at::parallel_for(0, B * P, grain_size, [&](int64_t start, int64_t end) {
                for (int64_t bp_idx = start; bp_idx < end; ++bp_idx) {
                    const int64_t b = bp_idx / P;
                    const int64_t p = bp_idx % P;
                    
                    BatchIndices indices(b, B_rot, shifts.has_value() ? shifts->size(0) : 1);
                    
                    for (int64_t i = 0; i < proj_boxsize; ++i) {
                        for (int64_t j = 0; j < proj_boxsize_half; ++j) {
                            // Convert to Fourier coordinates using shared utility
                            real_t proj_coord_r = index_to_fourier_coord<real_t>(i, proj_boxsize);
                            real_t proj_coord_c = j;

                            // Apply shared frequency filtering
                            if (should_filter_frequency<real_t>({proj_coord_r, proj_coord_c}, radius_cutoff_sq)) {
                                continue;
                            }
                            
                            // Apply oversampling and rotation
                            real_t sample_c = proj_coord_c * oversampling;
                            real_t sample_r = proj_coord_r * oversampling;
                            
                            // Apply 2x2 rotation matrix
                            real_t rot_r = rot_acc[indices.rot_b_idx][p][1][0] * sample_c + rot_acc[indices.rot_b_idx][p][1][1] * sample_r;
                            real_t rot_c = rot_acc[indices.rot_b_idx][p][0][0] * sample_c + rot_acc[indices.rot_b_idx][p][0][1] * sample_r;
                            
                            // Interpolate using shared kernel
                            scalar_t val = kernel.interpolate(rec_acc, b, boxsize, boxsize_half, rot_r, rot_c);
                            
                            // Apply phase shift using shared utility
                            if (shifts_acc.has_value()) {
                                std::vector<rot_real_t> shift_vals = {(*shifts_acc)[indices.shift_b_idx][p][0], 
                                                                      (*shifts_acc)[indices.shift_b_idx][p][1]};
                                std::vector<real_t> coord_vals = {proj_coord_r, proj_coord_c};
                                scalar_t phase_factor = compute_phase_factor<scalar_t, real_t, rot_real_t>(
                                    coord_vals, shift_vals, static_cast<real_t>(boxsize));
                                val = val * phase_factor;
                            }
                            
                            proj_acc[b][p][i][j] = val;
                        }
                    }
                }
            });
        }));
    }));

    return projection;
}

/**
 * Backward projection for 2D projections with smart gradient computation
 * 
 * Streamlined version using shared gradient setup and backward kernels.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> project_2d_back_cpu(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Use shared validation
    validate_projection_inputs(interpolation, reconstruction, rotations, shifts, 3, 2);
    TORCH_CHECK(grad_projections.is_complex(), "Projections must be a complex tensor");
    TORCH_CHECK(grad_projections.dim() == 4, "Projections must be a 4D tensor (B, P, boxsize, boxsize/2+1)");

    const auto B = grad_projections.size(0);
    const auto P = grad_projections.size(1);
    const auto proj_boxsize = grad_projections.size(2);
    const auto proj_boxsize_half = grad_projections.size(3);
    
    const auto rec_boxsize = reconstruction.size(1);
    const auto rec_boxsize_half = reconstruction.size(2);
    
    // Use shared gradient tensor setup
    GradientTensors grad_tensors(grad_projections, reconstruction, rotations, shifts);

    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "backward_project_2d_cpu", ([&] {
        using rot_real_t = scalar_t;
        auto rot_acc = rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        
        // Optional accessors for gradients
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 4, torch::DefaultPtrTraits>> grad_rot_acc;
        if (grad_tensors.need_rotation_grads) {
            grad_rot_acc.emplace(grad_tensors.grad_rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>());
        }
        
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> shifts_acc;
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> grad_shifts_acc;
        if (shifts.has_value()) {
            shifts_acc.emplace(shifts->packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
            if (grad_tensors.need_shift_grads) {
                grad_shifts_acc.emplace(grad_tensors.grad_shifts.packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
            }
        }

        AT_DISPATCH_COMPLEX_TYPES(grad_projections.scalar_type(), "backward_project_2d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;
            auto grad_proj_acc = grad_projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto grad_rec_acc = grad_tensors.grad_reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();
            auto rec_acc = reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();

            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;

            // Create kernels using shared components
            Interpolation2DKernel<scalar_t, real_t> kernel(interpolation);
            BackwardProjection2DKernel<scalar_t, real_t> backward_kernel(interpolation);
            Interpolation2DKernel<scalar_t, real_t> kernel_grad(interpolation);
            
            // Parallelize with shared grain size computation
            const int64_t grain_size = compute_grain_size(B * P);
            at::parallel_for(0, B * P, grain_size, [&](int64_t start, int64_t end) {
                for (int64_t bp_idx = start; bp_idx < end; ++bp_idx) {
                    const int64_t b = bp_idx / P;
                    const int64_t p = bp_idx % P;
                    
                    BatchIndices indices(b, rotations.size(0), shifts.has_value() ? shifts->size(0) : 1);
                    
                    // Local gradient accumulators
                    rot_real_t local_rot_grad[2][2] = {{0, 0}, {0, 0}};
                    rot_real_t local_shift_grad[2] = {0, 0};
                    
                    // Use shared gradient accumulator directly
                    
                    for (int64_t i = 0; i < proj_boxsize; ++i) {
                        for (int64_t j = 0; j < proj_boxsize_half; ++j) {
                            real_t proj_coord_r = index_to_fourier_coord<real_t>(i, proj_boxsize);
                            real_t proj_coord_c = j;

                            if (should_filter_frequency<real_t>({proj_coord_r, proj_coord_c}, radius_cutoff_sq)) {
                                continue;
                            }

                            if (j == 0 && i >= proj_boxsize / 2) {
                                // Skip Friedel-symmetric half of the x = 0 line (handled by other half)
                                continue;
                            }

                            real_t sample_c = proj_coord_c * oversampling;
                            real_t sample_r = proj_coord_r * oversampling;
                            
                            // Apply 2x2 rotation matrix
                            real_t rot_r = rot_acc[indices.rot_b_idx][p][1][0] * sample_c + rot_acc[indices.rot_b_idx][p][1][1] * sample_r;
                            real_t rot_c = rot_acc[indices.rot_b_idx][p][0][0] * sample_c + rot_acc[indices.rot_b_idx][p][0][1] * sample_r;
                            
                            scalar_t rec_val = kernel.interpolate(rec_acc, b, rec_boxsize, rec_boxsize_half, rot_r, rot_c);
                            scalar_t grad_proj = grad_proj_acc[b][p][i][j];

                            // Apply phase correction for shifts
                            if (shifts_acc.has_value()) {
                                std::vector<rot_real_t> shift_vals = {(*shifts_acc)[indices.shift_b_idx][p][0], 
                                                                      (*shifts_acc)[indices.shift_b_idx][p][1]};
                                std::vector<real_t> coord_vals = {-proj_coord_r, -proj_coord_c}; // Note: negative for backward
                                scalar_t phase_factor = compute_phase_factor<scalar_t, real_t, rot_real_t>(
                                    coord_vals, shift_vals, static_cast<real_t>(rec_boxsize));
                                grad_proj = grad_proj * phase_factor;
                            }

                            // Distribute gradients using shared backward kernel and accumulator
                            auto accumulate_func = [&](int64_t r, int64_t c, scalar_t grad) {
                                accumulate_2d_gradient(grad_rec_acc, b, rec_boxsize, rec_boxsize_half, r, c, grad);
                            };
                            backward_kernel.distribute_gradient(accumulate_func, grad_proj, rot_r, rot_c);

                            // Compute rotation gradients if needed
                            if (grad_tensors.need_rotation_grads) {
                                auto [rec_val_unused, grad_r, grad_c] = kernel_grad.interpolate_with_gradients(
                                    rec_acc, b, rec_boxsize, rec_boxsize_half, rot_r, rot_c);
                                
                                // Chain rule for rotation matrix gradients
                                local_rot_grad[0][0] += (grad_proj * std::conj(grad_c * sample_c)).real();
                                local_rot_grad[0][1] += (grad_proj * std::conj(grad_c * sample_r)).real();
                                local_rot_grad[1][0] += (grad_proj * std::conj(grad_r * sample_c)).real();
                                local_rot_grad[1][1] += (grad_proj * std::conj(grad_r * sample_r)).real();
                            }

                            // Compute shift gradients if needed
                            if (grad_tensors.need_shift_grads) {
                                scalar_t modulated_rec_val = rec_val;
                                if (shifts_acc.has_value()) {
                                    std::vector<rot_real_t> shift_vals = {(*shifts_acc)[indices.shift_b_idx][p][0], 
                                                                          (*shifts_acc)[indices.shift_b_idx][p][1]};
                                    std::vector<real_t> coord_vals = {proj_coord_r, proj_coord_c}; // Use positive coordinates for forward phase
                                    scalar_t phase_factor = compute_phase_factor<scalar_t, real_t, rot_real_t>(
                                        coord_vals, shift_vals, static_cast<real_t>(rec_boxsize));
                                    modulated_rec_val = rec_val * phase_factor;
                                }
                                
                                scalar_t phase_grad_r = scalar_t(0, -2.0 * M_PI * proj_coord_r / rec_boxsize) * modulated_rec_val;
                                scalar_t phase_grad_c = scalar_t(0, -2.0 * M_PI * proj_coord_c / rec_boxsize) * modulated_rec_val;
                                
                                scalar_t original_grad_proj = grad_proj_acc[b][p][i][j];
                                local_shift_grad[0] += (original_grad_proj * std::conj(phase_grad_r)).real();
                                local_shift_grad[1] += (original_grad_proj * std::conj(phase_grad_c)).real();
                            }
                        }
                    }
                    
                    // Atomically add local gradients to global tensors
                    if (grad_tensors.need_rotation_grads && grad_rot_acc.has_value()) {
                        atomic_add_real(&(*grad_rot_acc)[indices.rot_b_idx][p][0][0], local_rot_grad[0][0]);
                        atomic_add_real(&(*grad_rot_acc)[indices.rot_b_idx][p][0][1], local_rot_grad[0][1]);
                        atomic_add_real(&(*grad_rot_acc)[indices.rot_b_idx][p][1][0], local_rot_grad[1][0]);
                        atomic_add_real(&(*grad_rot_acc)[indices.rot_b_idx][p][1][1], local_rot_grad[1][1]);
                    }
                    
                    if (grad_tensors.need_shift_grads && grad_shifts_acc.has_value()) {
                        atomic_add_real(&(*grad_shifts_acc)[indices.shift_b_idx][p][0], local_shift_grad[0]);
                        atomic_add_real(&(*grad_shifts_acc)[indices.shift_b_idx][p][1], local_shift_grad[1]);
                    }
                }
            });
        }));
    }));

    return std::make_tuple(grad_tensors.grad_reconstruction, grad_tensors.grad_rotations, grad_tensors.grad_shifts);
}