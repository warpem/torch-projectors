/**
 * CPU Kernels for Differentiable 2D->3D Back-Projection Operations in Fourier Space
 * 
 * This file implements back-projection operations that accumulate 2D projections
 * into 3D reconstructions. This is the mathematical adjoint/transpose of 3D->2D
 * forward projection and follows the Central Slice Theorem.
 * 
 * Key Features:
 * - Back-projection: Accumulate 2D projection data into 3D Fourier reconstruction
 * - Full gradient support for projections, rotations, and shifts
 * - Uses shared interpolation kernels, FFTW sampling, and atomic operations
 * - Follows the Central Slice Theorem in Fourier space
 */

#include "backprojection_2d_to_3d_kernels.h"
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
 * 3D Backward projection kernel for gradient distribution
 * 
 * Wraps the common backward kernel interface to work with the legacy API
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
class BackwardProjection3DKernel {
private:
    std::unique_ptr<BackwardKernel<3, scalar_t, real_t>> kernel_;

public:
    BackwardProjection3DKernel(const std::string& interpolation) 
        : kernel_(get_3d_backward_kernel<scalar_t, real_t>(interpolation)) {}
    
    void distribute_gradient(
        std::function<void(int64_t, int64_t, int64_t, scalar_t)> accumulate_func,
        scalar_t grad_val,
        real_t d, real_t r, real_t c
    ) const {
        // Convert legacy API to new array-based API
        auto new_accumulate_func = [&accumulate_func](const std::array<int64_t, 3>& indices, scalar_t grad) {
            accumulate_func(indices[0], indices[1], indices[2], grad);
        };
        
        std::array<real_t, 3> coords = {d, r, c};
        kernel_->distribute_gradient(new_accumulate_func, grad_val, coords);
    }
};

/**
 * 3D Interpolation kernel wrapper
 * 
 * Wraps the common interpolation interface to work with the legacy 3D API
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
class Interpolation3DKernel {
private:
    std::unique_ptr<InterpolationKernel<3, scalar_t, real_t>> kernel_;

public:
    Interpolation3DKernel(const std::string& interpolation) 
        : kernel_(get_interpolation_kernel<3, scalar_t, real_t>(interpolation)) {}
    
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t d, real_t r, real_t c
    ) const {
        std::array<real_t, 3> coords = {d, r, c};
        return kernel_->interpolate(rec, b, boxsize, boxsize_half, coords);
    }
    
    std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t d, real_t r, real_t c
    ) const {
        std::array<real_t, 3> coords = {d, r, c};
        auto [val, grads] = kernel_->interpolate_with_gradients(rec, b, boxsize, boxsize_half, coords);
        return std::make_tuple(val, grads[0], grads[1], grads[2]);
    }
};

/**
 * Back-projection from 2D projections to 3D reconstructions
 * 
 * Accumulates 2D projection data (and optional weights) into 3D reconstructions using the Central Slice Theorem.
 * This is the adjoint/transpose operation of 3D->2D forward projection.
 */
std::tuple<at::Tensor, at::Tensor> backproject_2d_to_3d_forw_cpu(
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    // Use shared validation - projections are the input, rotations define 3D dimensionality (3x3)
    TORCH_CHECK(projections.is_complex(), "Projections must be a complex tensor");
    TORCH_CHECK(projections.dim() == 4, "Projections must be (B, P, height, width/2+1)");
    validate_projection_inputs(interpolation, projections, rotations, shifts, 4, 3);
    
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

    // For cubic volumes, depth = height = width, need to account for oversampling
    const auto rec_boxsize_raw = static_cast<int64_t>(std::ceil(proj_boxsize * oversampling));
    const auto rec_boxsize = rec_boxsize_raw + (rec_boxsize_raw % 2);  // Ensure even number
    const auto rec_boxsize_half = rec_boxsize / 2 + 1;
    const auto rec_depth = rec_boxsize;
    
    // Initialize output tensors - 3D reconstructions (B, D, H, W/2+1)
    auto data_reconstruction = torch::zeros({B, rec_depth, rec_boxsize, rec_boxsize_half}, projections.options());
    at::Tensor weight_reconstruction;
    bool has_weights = weights.has_value();
    
    if (has_weights) {
        weight_reconstruction = torch::zeros({B, rec_depth, rec_boxsize, rec_boxsize_half}, weights->options());
    }

    // PyTorch dispatch for type-generic code
    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "back_project_2d_to_3d_cpu_rotations", ([&] {
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
        AT_DISPATCH_COMPLEX_TYPES(projections.scalar_type(), "back_project_2d_to_3d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;
            auto proj_acc = projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto data_rec_acc = data_reconstruction.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();

            // Optional weight accessors
            c10::optional<torch::PackedTensorAccessor32<rot_real_t, 4, torch::DefaultPtrTraits>> weights_acc;
            c10::optional<torch::PackedTensorAccessor32<rot_real_t, 4, torch::DefaultPtrTraits>> weight_rec_acc;
            if (has_weights) {
                weights_acc.emplace(weights->packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>());
                weight_rec_acc.emplace(weight_reconstruction.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>());
            }

            // Set up Fourier space filtering
            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;

            // Create backward kernel using shared components
            BackwardProjection3DKernel<scalar_t, real_t> backward_kernel(interpolation);

            // Main back-projection loop with shared grain size computation
            const int64_t grain_size = compute_grain_size(B * P);
            at::parallel_for(0, B * P, grain_size, [&](int64_t start, int64_t end) {
                for (int64_t bp_idx = start; bp_idx < end; ++bp_idx) {
                    const int64_t b = bp_idx / P;
                    const int64_t p = bp_idx % P;
                    
                    BatchIndices indices(b, B_rot, shifts.has_value() ? shifts->size(0) : 1);
                    
                    for (int64_t i = 0; i < proj_boxsize; ++i) {
                        for (int64_t j = 0; j < proj_boxsize_half; ++j) {
                            // Convert to 2D Fourier coordinates
                            real_t proj_coord_r = index_to_fourier_coord<real_t>(i, proj_boxsize);
                            real_t proj_coord_c = j;

                            // Apply shared frequency filtering
                            if (should_filter_frequency<real_t>({proj_coord_r, proj_coord_c}, radius_cutoff_sq)) {
                                continue;
                            }

                            if (j == 0 && i >= proj_boxsize / 2) {
                                // Skip Friedel-symmetric half of the x = 0 line (handled by other half)
                                continue;
                            }
                            
                            // Apply oversampling to 2D coordinates
                            real_t sample_c = proj_coord_c * oversampling;
                            real_t sample_r = proj_coord_r * oversampling;
                            real_t sample_d = 0.0;  // Central slice through origin
                            
                            // Apply 3x3 rotation matrix to get 3D sampling coordinates
                            // This is the inverse transformation of the forward 3D->2D projection
                            real_t rot_c = rot_acc[indices.rot_b_idx][p][0][0] * sample_c + 
                                          rot_acc[indices.rot_b_idx][p][0][1] * sample_r + 
                                          rot_acc[indices.rot_b_idx][p][0][2] * sample_d;
                            real_t rot_r = rot_acc[indices.rot_b_idx][p][1][0] * sample_c + 
                                          rot_acc[indices.rot_b_idx][p][1][1] * sample_r + 
                                          rot_acc[indices.rot_b_idx][p][1][2] * sample_d;
                            real_t rot_d = rot_acc[indices.rot_b_idx][p][2][0] * sample_c + 
                                          rot_acc[indices.rot_b_idx][p][2][1] * sample_r + 
                                          rot_acc[indices.rot_b_idx][p][2][2] * sample_d;
                            
                            // Get projection data
                            scalar_t proj_val = proj_acc[b][p][i][j];
                            
                            // Apply conjugate phase shift for back-projection
                            if (shifts_acc.has_value()) {
                                std::vector<rot_real_t> shift_vals = {(*shifts_acc)[indices.shift_b_idx][p][0], 
                                                                      (*shifts_acc)[indices.shift_b_idx][p][1]};
                                std::vector<real_t> coord_vals = {proj_coord_r, proj_coord_c};
                                scalar_t phase_factor = compute_phase_factor<scalar_t, real_t, rot_real_t>(
                                    coord_vals, shift_vals, static_cast<real_t>(proj_boxsize));
                                // Use conjugate for back-projection
                                proj_val = proj_val * std::conj(phase_factor);
                            }
                            
                            // Distribute projection data to 3D reconstruction using backward kernel
                            auto data_accumulate_func = [&](int64_t d, int64_t r, int64_t c, scalar_t grad) {
                                accumulate_3d_data(data_rec_acc, b, rec_boxsize, rec_boxsize_half, d, r, c, grad);
                            };
                            backward_kernel.distribute_gradient(data_accumulate_func, proj_val, rot_d, rot_r, rot_c);
                            
                            // Distribute weights if provided (similar to 2D implementation)
                            if (has_weights) {
                                rot_real_t weight_val = (*weights_acc)[b][p][i][j];
                                auto weight_accumulate_func = [&](int64_t d, int64_t r, int64_t c, scalar_t interp_weight) {
                                    // For weights, we only need the interpolation weight magnitude (absolute value)
                                    rot_real_t final_weight = weight_val * std::abs(interp_weight.real());
                                    // Handle bounds and 3D Friedel symmetry for weight accumulation
                                    int64_t d_eff = d;
                                    int64_t r_eff = r;
                                    int64_t c_eff = c;
                                    
                                    // Handle negative frequencies with Friedel symmetry
                                    if (c_eff < 0) { 
                                        c_eff = -c_eff;
                                        r_eff = -r_eff;
                                        d_eff = -d_eff;
                                    }
                                    
                                    // Bounds checking for 3D
                                    if (c_eff >= rec_boxsize_half) return;
                                    if (d_eff > rec_depth / 2 || d_eff < -rec_depth / 2 + 1) return;
                                    if (r_eff > rec_boxsize / 2 || r_eff < -rec_boxsize / 2 + 1) return;
                                    
                                    // Convert negative indices
                                    int64_t d_idx = d_eff < 0 ? rec_depth + d_eff : d_eff;
                                    int64_t r_idx = r_eff < 0 ? rec_boxsize + r_eff : r_eff;
                                    
                                    if (d_idx >= rec_depth || r_idx >= rec_boxsize) return;
                                    
                                    atomic_add_real(&(*weight_rec_acc)[b][d_idx][r_idx][c_eff], final_weight);

                                    // On the x=0 line, also insert Friedel-symmetric conjugate counterpart
                                    if (c == 0) {
                                        int64_t r_eff2 = -1 * r_eff;
                                        int64_t d_eff2 = -1 * d_eff;
                                        d_eff2 = d_eff2 < 0 ? rec_depth + d_eff2 : d_eff2;
                                        r_eff2 = r_eff2 < 0 ? rec_boxsize + r_eff2 : r_eff2;
                                        if (d_eff2 >= rec_depth || 
                                            r_eff2 >= rec_boxsize || 
                                            (r_eff2 == r_eff && d_eff2 == d_eff)) return;

                                        atomic_add_real(&(*weight_rec_acc)[b][d_eff2][r_eff2][c], final_weight);
                                    }
                                };
                                backward_kernel.distribute_gradient(weight_accumulate_func, scalar_t(1.0, 0.0), rot_d, rot_r, rot_c);
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
 * Backward pass for 2D->3D back-projection with gradient computation
 * 
 * Computes gradients w.r.t. projections, weights, rotations, and shifts.
 * The key insight is that grad_projections = forward_project_3d_to_2d(grad_data_rec) 
 * since 3D->2D forward projection is the mathematical adjoint of 2D->3D back-projection.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backproject_2d_to_3d_back_cpu(
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
    TORCH_CHECK(grad_data_rec.dim() == 4, "grad_data_rec must be (B, D, H, W/2+1)");
    validate_projection_inputs(interpolation, projections, rotations, shifts, 4, 3);
    
    if (grad_weight_rec.has_value()) {
        TORCH_CHECK(grad_weight_rec->is_floating_point(), "grad_weight_rec must be a real-valued tensor");
        TORCH_CHECK(grad_weight_rec->dim() == 4, "grad_weight_rec must be (B, D, H, W/2+1)");
        TORCH_CHECK(grad_weight_rec->sizes() == grad_data_rec.sizes(), "grad_weight_rec and grad_data_rec must have same shape");
    }
    
    const auto B = projections.size(0);
    const auto P = projections.size(1);
    const auto proj_boxsize = projections.size(2);
    const auto proj_boxsize_half = projections.size(3);
    
    // Reconstruction size should match what was used in forward pass
    const auto rec_boxsize_raw = static_cast<int64_t>(std::ceil(proj_boxsize * oversampling));
    const auto rec_boxsize = rec_boxsize_raw + (rec_boxsize_raw % 2);  // Ensure even number
    const auto rec_boxsize_half = rec_boxsize / 2 + 1;
    const auto rec_depth = rec_boxsize;
    
    // Validate that grad_data_rec has the expected dimensions
    TORCH_CHECK(grad_data_rec.size(1) == rec_depth, "grad_data_rec depth must match expected reconstruction size");
    TORCH_CHECK(grad_data_rec.size(2) == rec_boxsize, "grad_data_rec height must match expected reconstruction size");
    TORCH_CHECK(grad_data_rec.size(3) == rec_boxsize_half, "grad_data_rec width must match expected reconstruction size");
    
    // Initialize gradient tensors
    auto grad_projections = torch::zeros_like(projections);
    at::Tensor grad_weights;
    if (weights.has_value()) {
        grad_weights = torch::zeros_like(*weights);
    } else {
        grad_weights = torch::empty({0}, projections.options().dtype(rotations.scalar_type()));
    }
    
    // Setup gradient tensors for rotations and shifts using shared utility
    GradientTensors<at::Tensor> grad_tensors(grad_projections, grad_data_rec, rotations, shifts);
    
    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "backward_back_project_2d_to_3d_cpu", ([&] {
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

        AT_DISPATCH_COMPLEX_TYPES(projections.scalar_type(), "backward_back_project_2d_to_3d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;
            auto grad_rec_acc = grad_data_rec.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto grad_proj_acc = grad_projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto proj_acc = projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();

            // Optional weight accessors
            c10::optional<torch::PackedTensorAccessor32<rot_real_t, 4, torch::DefaultPtrTraits>> weights_acc;
            c10::optional<torch::PackedTensorAccessor32<rot_real_t, 4, torch::DefaultPtrTraits>> grad_weights_acc;
            c10::optional<torch::PackedTensorAccessor32<rot_real_t, 4, torch::DefaultPtrTraits>> grad_weight_rec_acc;
            
            if (weights.has_value()) {
                weights_acc.emplace(weights->packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>());
                grad_weights_acc.emplace(grad_weights.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>());
            }
            if (grad_weight_rec.has_value()) {
                grad_weight_rec_acc.emplace(grad_weight_rec->packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>());
            }

            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;

            // Create kernels using shared components
            Interpolation3DKernel<scalar_t, real_t> kernel(interpolation);
            Interpolation3DKernel<scalar_t, real_t> kernel_grad(interpolation);
            
            // Parallelize with shared grain size computation
            const int64_t grain_size = compute_grain_size(B * P);
            at::parallel_for(0, B * P, grain_size, [&](int64_t start, int64_t end) {
                for (int64_t bp_idx = start; bp_idx < end; ++bp_idx) {
                    const int64_t b = bp_idx / P;
                    const int64_t p = bp_idx % P;
                    
                    BatchIndices indices(b, rotations.size(0), shifts.has_value() ? shifts->size(0) : 1);
                    
                    // Local gradient accumulators for 3x3 rotation matrix
                    rot_real_t local_rot_grad[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                    rot_real_t local_shift_grad[2] = {0, 0};
                    
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
                            real_t sample_d = 0.0;  // Central slice
                            
                            // Apply 3x3 rotation matrix
                            real_t rot_c = rot_acc[indices.rot_b_idx][p][0][0] * sample_c + 
                                          rot_acc[indices.rot_b_idx][p][0][1] * sample_r + 
                                          rot_acc[indices.rot_b_idx][p][0][2] * sample_d;
                            real_t rot_r = rot_acc[indices.rot_b_idx][p][1][0] * sample_c + 
                                          rot_acc[indices.rot_b_idx][p][1][1] * sample_r + 
                                          rot_acc[indices.rot_b_idx][p][1][2] * sample_d;
                            real_t rot_d = rot_acc[indices.rot_b_idx][p][2][0] * sample_c + 
                                          rot_acc[indices.rot_b_idx][p][2][1] * sample_r + 
                                          rot_acc[indices.rot_b_idx][p][2][2] * sample_d;
                            
                            // Compute grad_projections using forward projection (adjoint relationship)
                            scalar_t rec_val = kernel.interpolate(grad_rec_acc, b, rec_boxsize, rec_boxsize_half, rot_d, rot_r, rot_c);
                            
                            // Apply phase shift (opposite of back-projection conjugate)
                            if (shifts_acc.has_value()) {
                                std::vector<rot_real_t> shift_vals = {(*shifts_acc)[indices.shift_b_idx][p][0], 
                                                                      (*shifts_acc)[indices.shift_b_idx][p][1]};
                                std::vector<real_t> coord_vals = {proj_coord_r, proj_coord_c};
                                scalar_t phase_factor = compute_phase_factor<scalar_t, real_t, rot_real_t>(
                                    coord_vals, shift_vals, static_cast<real_t>(proj_boxsize));
                                rec_val = rec_val * phase_factor;  // Forward phase for grad_projections
                            }
                            
                            grad_proj_acc[b][p][i][j] = rec_val;

                            // Compute grad_weights if needed
                            if (grad_weight_rec_acc.has_value() && grad_weights_acc.has_value()) {
                                // Use trilinear interpolation for 3D real-valued gradient
                                const int64_t c_floor = floor(rot_c);
                                const int64_t r_floor = floor(rot_r);
                                const int64_t d_floor = floor(rot_d);
                                const real_t c_frac = rot_c - c_floor;
                                const real_t r_frac = rot_r - r_floor;
                                const real_t d_frac = rot_d - d_floor;

                                // Sample 2x2x2 grid from grad_weight_rec with bounds checking
                                auto sample_weight_grad = [&](int64_t d, int64_t r, int64_t c) -> rot_real_t {
                                    // Handle Friedel symmetry and bounds for 3D
                                    if (c < 0) { 
                                        c = -c;
                                        r = -r;
                                        d = -d;
                                    }
                                    if (c >= rec_boxsize_half) return 0.0;
                                    if (d > rec_depth / 2 || d < -rec_depth / 2 + 1) return 0.0;
                                    if (r > rec_boxsize / 2 || r < -rec_boxsize / 2 + 1) return 0.0;
                                    
                                    int64_t d_eff = d < 0 ? rec_depth + d : d;
                                    int64_t r_eff = r < 0 ? rec_boxsize + r : r;
                                    if (d_eff >= rec_depth || r_eff >= rec_boxsize) return 0.0;
                                    
                                    return (*grad_weight_rec_acc)[b][d_eff][r_eff][c];
                                };

                                // Sample all 8 corners of the 3D cube
                                const rot_real_t p000 = sample_weight_grad(d_floor, r_floor, c_floor);
                                const rot_real_t p001 = sample_weight_grad(d_floor, r_floor, c_floor + 1);
                                const rot_real_t p010 = sample_weight_grad(d_floor, r_floor + 1, c_floor);
                                const rot_real_t p011 = sample_weight_grad(d_floor, r_floor + 1, c_floor + 1);
                                const rot_real_t p100 = sample_weight_grad(d_floor + 1, r_floor, c_floor);
                                const rot_real_t p101 = sample_weight_grad(d_floor + 1, r_floor, c_floor + 1);
                                const rot_real_t p110 = sample_weight_grad(d_floor + 1, r_floor + 1, c_floor);
                                const rot_real_t p111 = sample_weight_grad(d_floor + 1, r_floor + 1, c_floor + 1);

                                // Trilinear interpolation
                                const rot_real_t p00 = p000 + (p001 - p000) * c_frac;
                                const rot_real_t p01 = p010 + (p011 - p010) * c_frac;
                                const rot_real_t p10 = p100 + (p101 - p100) * c_frac;
                                const rot_real_t p11 = p110 + (p111 - p110) * c_frac;
                                
                                const rot_real_t p0 = p00 + (p01 - p00) * r_frac;
                                const rot_real_t p1 = p10 + (p11 - p10) * r_frac;
                                const rot_real_t weight_grad = p0 + (p1 - p0) * d_frac;
                                
                                (*grad_weights_acc)[b][p][i][j] = weight_grad;
                            }

                            // Compute 3x3 rotation gradients if needed
                            if (grad_tensors.need_rotation_grads) {
                                auto [rec_val_unused, grad_d, grad_r, grad_c] = kernel_grad.interpolate_with_gradients(
                                    grad_rec_acc, b, rec_boxsize, rec_boxsize_half, rot_d, rot_r, rot_c);
                                
                                scalar_t proj_val = proj_acc[b][p][i][j];
                                
                                // Apply conjugate phase shift to projection value
                                if (shifts_acc.has_value()) {
                                    std::vector<rot_real_t> shift_vals = {(*shifts_acc)[indices.shift_b_idx][p][0], 
                                                                          (*shifts_acc)[indices.shift_b_idx][p][1]};
                                    std::vector<real_t> coord_vals = {proj_coord_r, proj_coord_c};
                                    scalar_t phase_factor = compute_phase_factor<scalar_t, real_t, rot_real_t>(
                                        coord_vals, shift_vals, static_cast<real_t>(proj_boxsize));
                                    proj_val = proj_val * std::conj(phase_factor);
                                }
                                
                                // Chain rule for 3x3 rotation matrix gradients
                                // [rot_c; rot_r; rot_d] = R * [sample_c; sample_r; sample_d]
                                local_rot_grad[0][0] += (proj_val * std::conj(grad_c * sample_c)).real();  // ∂f/∂R[0][0]
                                local_rot_grad[0][1] += (proj_val * std::conj(grad_c * sample_r)).real();  // ∂f/∂R[0][1]
                                local_rot_grad[0][2] += (proj_val * std::conj(grad_c * sample_d)).real();  // ∂f/∂R[0][2]
                                local_rot_grad[1][0] += (proj_val * std::conj(grad_r * sample_c)).real();  // ∂f/∂R[1][0]
                                local_rot_grad[1][1] += (proj_val * std::conj(grad_r * sample_r)).real();  // ∂f/∂R[1][1]
                                local_rot_grad[1][2] += (proj_val * std::conj(grad_r * sample_d)).real();  // ∂f/∂R[1][2]
                                local_rot_grad[2][0] += (proj_val * std::conj(grad_d * sample_c)).real();  // ∂f/∂R[2][0]
                                local_rot_grad[2][1] += (proj_val * std::conj(grad_d * sample_r)).real();  // ∂f/∂R[2][1]
                                local_rot_grad[2][2] += (proj_val * std::conj(grad_d * sample_d)).real();  // ∂f/∂R[2][2]
                            }

                            // Compute shift gradients if needed
                            if (grad_tensors.need_shift_grads) {
                                scalar_t rec_val_for_shift = kernel.interpolate(grad_rec_acc, b, rec_boxsize, rec_boxsize_half, rot_d, rot_r, rot_c);
                                scalar_t proj_val = proj_acc[b][p][i][j];
                                
                                // Apply conjugate phase shift to projection value (consistent with rotation gradient computation)
                                if (shifts_acc.has_value()) {
                                    std::vector<rot_real_t> shift_vals = {(*shifts_acc)[indices.shift_b_idx][p][0], 
                                                                          (*shifts_acc)[indices.shift_b_idx][p][1]};
                                    std::vector<real_t> coord_vals = {proj_coord_r, proj_coord_c};
                                    scalar_t phase_factor = compute_phase_factor<scalar_t, real_t, rot_real_t>(
                                        coord_vals, shift_vals, static_cast<real_t>(proj_boxsize));
                                    proj_val = proj_val * std::conj(phase_factor);
                                }
                                
                                scalar_t phase_grad_r = scalar_t(0, -2.0 * M_PI * proj_coord_r / proj_boxsize) * rec_val_for_shift;
                                scalar_t phase_grad_c = scalar_t(0, -2.0 * M_PI * proj_coord_c / proj_boxsize) * rec_val_for_shift;
                                
                                local_shift_grad[0] += (proj_val * std::conj(phase_grad_r)).real();
                                local_shift_grad[1] += (proj_val * std::conj(phase_grad_c)).real();
                            }
                        }
                    }
                    
                    // Atomically add local gradients to global tensors
                    if (grad_tensors.need_rotation_grads && grad_rot_acc.has_value()) {
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                atomic_add_real(&(*grad_rot_acc)[indices.rot_b_idx][p][i][j], local_rot_grad[i][j]);
                            }
                        }
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