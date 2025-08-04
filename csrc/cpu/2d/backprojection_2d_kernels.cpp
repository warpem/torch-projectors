/**
 * CPU Kernels for Differentiable 2D Back-Projection Operations in Fourier Space
 * 
 * This file implements back-projection operations that accumulate 2D projections
 * into 2D reconstructions. This is the mathematical adjoint/transpose of forward
 * projection and supports optional weight accumulation for CTF handling.
 * 
 * Key Features:
 * - Back-projection: Accumulate 2D projection data into 2D Fourier reconstruction
 * - Optional weight accumulation for CTF^2 or similar applications
 * - Full gradient support for projections, weights, rotations, and shifts
 * - Uses shared interpolation kernels, FFTW sampling, and atomic operations
 * - Follows the Projection-Slice Theorem in Fourier space
 */

#include "backprojection_2d_kernels.h"
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
        : kernel_(get_backward_kernel<2, scalar_t, real_t>(interpolation)) {}
    
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
 * Back-projection from 2D projections to 2D reconstructions
 * 
 * Accumulates 2D projection data (and optional weights) into 2D reconstructions.
 * This is the adjoint/transpose operation of forward projection.
 */
std::tuple<at::Tensor, at::Tensor> backproject_2d_forw_cpu(
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Use shared validation - projections are the "reconstruction" input, rotations define dimensionality
    TORCH_CHECK(projections.is_complex(), "Projections must be a complex tensor");
    TORCH_CHECK(projections.dim() == 4, "Projections must be (B, P, height, width/2+1)");
    validate_projection_inputs(interpolation, projections, rotations, shifts, 4, 2);
    
    // Validate optional weights
    if (weights.has_value()) {
        TORCH_CHECK(weights->is_floating_point(), "Weights must be a real-valued tensor");
        TORCH_CHECK(weights->dim() == 4, "Weights must be (B, P, height, width/2+1)");
        TORCH_CHECK(weights->sizes() == projections.sizes(), "Weights and projections must have the same shape");
    }

    // Extract tensor dimensions
    const auto B = projections.size(0);
    const auto P = projections.size(1); 
    const auto proj_boxsize = projections.size(2);
    const auto proj_boxsize_half = projections.size(3);
    
    const auto B_rot = rotations.size(0);
    TORCH_CHECK(B_rot == B || B_rot == 1, "Batch size of rotations must be 1 or same as projections");

    // Reconstruction size matches projection size
    const auto rec_boxsize = proj_boxsize;
    const auto rec_boxsize_half = proj_boxsize_half;
    
    // Initialize output tensors
    auto data_reconstruction = torch::zeros({B, rec_boxsize, rec_boxsize_half}, projections.options());
    at::Tensor weight_reconstruction;
    bool has_weights = weights.has_value();
    
    if (has_weights) {
        weight_reconstruction = torch::zeros({B, rec_boxsize, rec_boxsize_half}, weights->options());
    }

    // PyTorch dispatch for type-generic code
    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "back_project_2d_cpu_rotations", ([&] {
        using rot_real_t = scalar_t;
        auto rot_acc = rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        
        // Handle optional shifts
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> shifts_acc;
        if (shifts.has_value()) {
            TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
            TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1, "Batch size of shifts must be 1 or same as projections");
            TORCH_CHECK(shifts->size(1) == P, "Number of poses in shifts must match rotations");
            TORCH_CHECK(shifts->scalar_type() == rotations.scalar_type(), "Shifts and rotations must have the same dtype");
            shifts_acc.emplace(shifts->packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
        }
        
        // Dispatch for complex types
        AT_DISPATCH_COMPLEX_TYPES(projections.scalar_type(), "back_project_2d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;
            auto proj_acc = projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto data_rec_acc = data_reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();

            // Optional weight accessors
            c10::optional<torch::PackedTensorAccessor32<rot_real_t, 4, torch::DefaultPtrTraits>> weights_acc;
            c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> weight_rec_acc;
            if (has_weights) {
                weights_acc.emplace(weights->packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>());
                weight_rec_acc.emplace(weight_reconstruction.packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
            }

            // Set up Fourier space filtering
            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;

            // Create backward kernel using shared components
            BackwardProjection2DKernel<scalar_t, real_t> backward_kernel(interpolation);

            // Main back-projection loop with shared grain size computation
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
                            
                            // Apply oversampling and rotation (same as forward projection)
                            real_t sample_c = proj_coord_c * oversampling;
                            real_t sample_r = proj_coord_r * oversampling;
                            
                            // Apply 2x2 rotation matrix (same as forward projection)
                            real_t rot_r = rot_acc[indices.rot_b_idx][p][1][0] * sample_c + rot_acc[indices.rot_b_idx][p][1][1] * sample_r;
                            real_t rot_c = rot_acc[indices.rot_b_idx][p][0][0] * sample_c + rot_acc[indices.rot_b_idx][p][0][1] * sample_r;
                            
                            // Get projection data
                            scalar_t proj_val = proj_acc[b][p][i][j];
                            
                            // Apply conjugate phase shift for back-projection
                            if (shifts_acc.has_value()) {
                                std::vector<rot_real_t> shift_vals = {(*shifts_acc)[indices.shift_b_idx][p][0], 
                                                                      (*shifts_acc)[indices.shift_b_idx][p][1]};
                                std::vector<real_t> coord_vals = {proj_coord_r, proj_coord_c};
                                scalar_t phase_factor = compute_phase_factor<scalar_t, real_t, rot_real_t>(
                                    coord_vals, shift_vals, static_cast<real_t>(rec_boxsize));
                                // Use conjugate for back-projection
                                proj_val = proj_val * std::conj(phase_factor);
                            }
                            
                            // Distribute projection data to reconstruction using backward kernel
                            auto data_accumulate_func = [&](int64_t r, int64_t c, scalar_t grad) {
                                accumulate_2d_gradient(data_rec_acc, b, rec_boxsize, rec_boxsize_half, r, c, grad);
                            };
                            backward_kernel.distribute_gradient(data_accumulate_func, proj_val, rot_r, rot_c);
                            
                            // Distribute weights if provided
                            if (has_weights) {
                                rot_real_t weight_val = (*weights_acc)[b][p][i][j];
                                auto weight_accumulate_func = [&](int64_t r, int64_t c, scalar_t interp_weight) {
                                    // For weights, we only need the interpolation weight magnitude (absolute value)
                                    rot_real_t final_weight = weight_val * std::abs(interp_weight.real());
                                    // Handle bounds and Friedel symmetry for weight accumulation
                                    bool needs_conj = false;
                                    int64_t c_eff = c;
                                    int64_t r_eff = r;
                                    
                                    if (c_eff < 0) { 
                                        c_eff = -c_eff;
                                        r_eff = -r_eff;
                                        needs_conj = true;
                                    }
                                    
                                    if (c_eff >= rec_boxsize_half) return;
                                    if (r_eff > rec_boxsize / 2 || r_eff < -rec_boxsize / 2 + 1) return;
                                    
                                    int64_t r_idx = r_eff < 0 ? rec_boxsize + r_eff : r_eff;
                                    if (r_idx >= rec_boxsize) return;
                                    
                                    atomic_add_real(&(*weight_rec_acc)[b][r_idx][c_eff], final_weight);
                                };
                                backward_kernel.distribute_gradient(weight_accumulate_func, scalar_t(1.0, 0.0), rot_r, rot_c);
                            }
                        }
                    }
                }
            });
        }));
    }));

    // Return appropriate result based on whether weights were provided
    if (has_weights) {
        return std::make_tuple(data_reconstruction, weight_reconstruction);
    } else {
        // Return empty tensor for weight_reconstruction when weights not provided
        at::Tensor empty_weights = torch::empty({0}, data_reconstruction.options().dtype(rotations.scalar_type()));
        return std::make_tuple(data_reconstruction, empty_weights);
    }
}

/**
 * Backward pass for 2D back-projection with gradient computation
 * 
 * Computes gradients w.r.t. projections, weights, rotations, and shifts.
 * The key insight is that grad_projections = forward_project(grad_data_rec) since
 * forward projection is the mathematical adjoint of back-projection.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backproject_2d_back_cpu(
    const at::Tensor& grad_data_rec,
    const c10::optional<at::Tensor>& grad_weight_rec,
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Validate inputs
    TORCH_CHECK(grad_data_rec.is_complex(), "grad_data_rec must be a complex tensor");
    TORCH_CHECK(grad_data_rec.dim() == 3, "grad_data_rec must be (B, height, width/2+1)");
    validate_projection_inputs(interpolation, projections, rotations, shifts, 4, 2);
    
    if (grad_weight_rec.has_value()) {
        TORCH_CHECK(grad_weight_rec->is_floating_point(), "grad_weight_rec must be a real-valued tensor");
        TORCH_CHECK(grad_weight_rec->dim() == 3, "grad_weight_rec must be (B, height, width/2+1)");
        TORCH_CHECK(grad_weight_rec->sizes() == grad_data_rec.sizes(), "grad_weight_rec and grad_data_rec must have same shape");
    }
    
    const auto B = projections.size(0);
    const auto P = projections.size(1);
    const auto proj_boxsize = projections.size(2);
    const auto proj_boxsize_half = projections.size(3);
    
    const auto rec_boxsize = grad_data_rec.size(1);
    const auto rec_boxsize_half = grad_data_rec.size(2);
    
    // Initialize gradient tensors
    auto grad_projections = torch::zeros_like(projections);
    at::Tensor grad_weights;
    if (weights.has_value()) {
        grad_weights = torch::zeros_like(*weights);
    } else {
        grad_weights = torch::empty({0}, projections.options().dtype(rotations.scalar_type()));
    }
    
    // Setup gradient tensors for rotations and shifts using shared utility
    GradientTensors<at::Tensor> grad_tensors(projections, grad_data_rec, rotations, shifts);
    
    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "backward_back_project_2d_cpu", ([&] {
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

        AT_DISPATCH_COMPLEX_TYPES(projections.scalar_type(), "backward_back_project_2d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;
            auto grad_data_rec_acc = grad_data_rec.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();
            auto grad_proj_acc = grad_projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto proj_acc = projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();

            // Optional weight accessors
            c10::optional<torch::PackedTensorAccessor32<rot_real_t, 4, torch::DefaultPtrTraits>> weights_acc;
            c10::optional<torch::PackedTensorAccessor32<rot_real_t, 4, torch::DefaultPtrTraits>> grad_weights_acc;
            c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> grad_weight_rec_acc;
            
            if (weights.has_value()) {
                weights_acc.emplace(weights->packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>());
                grad_weights_acc.emplace(grad_weights.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>());
            }
            if (grad_weight_rec.has_value()) {
                grad_weight_rec_acc.emplace(grad_weight_rec->packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
            }

            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;

            // Create kernels using shared components
            Interpolation2DKernel<scalar_t, real_t> kernel(interpolation);
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
                    
                    for (int64_t i = 0; i < proj_boxsize; ++i) {
                        for (int64_t j = 0; j < proj_boxsize_half; ++j) {
                            real_t proj_coord_r = index_to_fourier_coord<real_t>(i, proj_boxsize);
                            real_t proj_coord_c = j;

                            if (should_filter_frequency<real_t>({proj_coord_r, proj_coord_c}, radius_cutoff_sq)) {
                                continue;
                            }

                            real_t sample_c = proj_coord_c * oversampling;
                            real_t sample_r = proj_coord_r * oversampling;
                            
                            // Apply 2x2 rotation matrix
                            real_t rot_r = rot_acc[indices.rot_b_idx][p][1][0] * sample_c + rot_acc[indices.rot_b_idx][p][1][1] * sample_r;
                            real_t rot_c = rot_acc[indices.rot_b_idx][p][0][0] * sample_c + rot_acc[indices.rot_b_idx][p][0][1] * sample_r;
                            
                            // Compute grad_projections using forward projection (adjoint relationship)
                            scalar_t rec_val = kernel.interpolate(grad_data_rec_acc, b, rec_boxsize, rec_boxsize_half, rot_r, rot_c);
                            
                            // Apply conjugate phase shift (opposite of back-projection)
                            if (shifts_acc.has_value()) {
                                std::vector<rot_real_t> shift_vals = {(*shifts_acc)[indices.shift_b_idx][p][0], 
                                                                      (*shifts_acc)[indices.shift_b_idx][p][1]};
                                std::vector<real_t> coord_vals = {proj_coord_r, proj_coord_c};
                                scalar_t phase_factor = compute_phase_factor<scalar_t, real_t, rot_real_t>(
                                    coord_vals, shift_vals, static_cast<real_t>(rec_boxsize));
                                rec_val = rec_val * phase_factor;  // Forward phase for grad_projections
                            }
                            
                            grad_proj_acc[b][p][i][j] = rec_val;

                            // Compute grad_weights if needed
                            if (grad_weight_rec_acc.has_value() && grad_weights_acc.has_value()) {
                                // Use linear interpolation for real-valued gradient
                                const int64_t c_floor = floor(rot_c);
                                const int64_t r_floor = floor(rot_r);
                                const real_t c_frac = rot_c - c_floor;
                                const real_t r_frac = rot_r - r_floor;

                                // Sample 2x2 grid from grad_weight_rec with bounds checking
                                auto sample_weight_grad = [&](int64_t r, int64_t c) -> rot_real_t {
                                    // Handle Friedel symmetry and bounds
                                    bool needs_conj = false;
                                    if (c < 0) { 
                                        c = -c;
                                        r = -r;
                                        needs_conj = true;
                                    }
                                    if (c >= rec_boxsize_half) return 0.0;
                                    if (r > rec_boxsize / 2 || r < -rec_boxsize / 2 + 1) return 0.0;
                                    
                                    int64_t r_eff = r < 0 ? rec_boxsize + r : r;
                                    if (r_eff >= rec_boxsize) return 0.0;
                                    
                                    return (*grad_weight_rec_acc)[b][r_eff][c];
                                };

                                const rot_real_t p00 = sample_weight_grad(r_floor, c_floor);
                                const rot_real_t p01 = sample_weight_grad(r_floor, c_floor + 1);
                                const rot_real_t p10 = sample_weight_grad(r_floor + 1, c_floor);
                                const rot_real_t p11 = sample_weight_grad(r_floor + 1, c_floor + 1);

                                // Bilinear interpolation
                                const rot_real_t p0 = p00 + (p01 - p00) * c_frac;
                                const rot_real_t p1 = p10 + (p11 - p10) * c_frac;
                                const rot_real_t weight_grad = p0 + (p1 - p0) * r_frac;
                                
                                (*grad_weights_acc)[b][p][i][j] = weight_grad;
                            }

                            // Compute rotation gradients if needed
                            if (grad_tensors.need_rotation_grads) {
                                auto [rec_val_unused, grad_r, grad_c] = kernel_grad.interpolate_with_gradients(
                                    grad_data_rec_acc, b, rec_boxsize, rec_boxsize_half, rot_r, rot_c);
                                
                                scalar_t proj_val = proj_acc[b][p][i][j];
                                
                                // Apply conjugate phase shift to projection value
                                if (shifts_acc.has_value()) {
                                    std::vector<rot_real_t> shift_vals = {(*shifts_acc)[indices.shift_b_idx][p][0], 
                                                                          (*shifts_acc)[indices.shift_b_idx][p][1]};
                                    std::vector<real_t> coord_vals = {proj_coord_r, proj_coord_c};
                                    scalar_t phase_factor = compute_phase_factor<scalar_t, real_t, rot_real_t>(
                                        coord_vals, shift_vals, static_cast<real_t>(rec_boxsize));
                                    proj_val = proj_val * std::conj(phase_factor);
                                }
                                
                                // Chain rule for rotation matrix gradients
                                local_rot_grad[0][0] += (proj_val * std::conj(grad_c * sample_c)).real();
                                local_rot_grad[0][1] += (proj_val * std::conj(grad_c * sample_r)).real();
                                local_rot_grad[1][0] += (proj_val * std::conj(grad_r * sample_c)).real();
                                local_rot_grad[1][1] += (proj_val * std::conj(grad_r * sample_r)).real();
                            }

                            // Compute shift gradients if needed
                            if (grad_tensors.need_shift_grads) {
                                scalar_t rec_val_for_shift = kernel.interpolate(grad_data_rec_acc, b, rec_boxsize, rec_boxsize_half, rot_r, rot_c);
                                scalar_t proj_val = proj_acc[b][p][i][j];
                                
                                // Apply conjugate phase shift to projection value (consistent with rotation gradient computation)
                                if (shifts_acc.has_value()) {
                                    std::vector<rot_real_t> shift_vals = {(*shifts_acc)[indices.shift_b_idx][p][0], 
                                                                          (*shifts_acc)[indices.shift_b_idx][p][1]};
                                    std::vector<real_t> coord_vals = {proj_coord_r, proj_coord_c};
                                    scalar_t phase_factor = compute_phase_factor<scalar_t, real_t, rot_real_t>(
                                        coord_vals, shift_vals, static_cast<real_t>(rec_boxsize));
                                    proj_val = proj_val * std::conj(phase_factor);
                                }
                                
                                scalar_t phase_grad_r = scalar_t(0, -2.0 * M_PI * proj_coord_r / rec_boxsize) * rec_val_for_shift;
                                scalar_t phase_grad_c = scalar_t(0, -2.0 * M_PI * proj_coord_c / rec_boxsize) * rec_val_for_shift;
                                
                                local_shift_grad[0] += (proj_val * std::conj(phase_grad_r)).real();
                                local_shift_grad[1] += (proj_val * std::conj(phase_grad_c)).real();
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

    return std::make_tuple(grad_projections, grad_weights, grad_tensors.grad_rotations, grad_tensors.grad_shifts);
}