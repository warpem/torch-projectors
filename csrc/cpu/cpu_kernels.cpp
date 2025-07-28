#include "cpu_kernels.h"
#include <torch/extension.h>
#include <complex>
#include <algorithm>

// Helper function for sampling from FFTW-formatted Fourier space
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline scalar_t sample_fftw_with_conjugate(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
    const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
    int64_t r, int64_t c
) {
    bool need_conjugate = false;

    // Handle negative kx via Friedel symmetry (c < 0)
    if (c < 0) {
        c = -c;          // mirror to positive kx
        r = -r;          // ky must be mirrored as well
        need_conjugate = !need_conjugate;
    }

    c = std::min(c, boxsize_half - 1);
    r = std::min(boxsize / 2, std::max(r, -boxsize / 2 + 1));

    if (r < 0)
        r = boxsize + r;

    r = std::min(r, boxsize - 1);

    if (need_conjugate)
        return std::conj(rec[b][r][c]);
    else
        return rec[b][r][c];
}



// Interpolation method enumeration
enum class InterpolationMethod {
    LINEAR,
    CUBIC
};

// Abstract base for interpolation kernels
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct InterpolationKernel {
    virtual ~InterpolationKernel() = default;
    
    virtual scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const = 0;
    
    virtual std::tuple<scalar_t, scalar_t, scalar_t> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const = 0;
};

// Bilinear interpolation kernel
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct BilinearKernel : public InterpolationKernel<scalar_t, real_t> {
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const override {
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);

        const real_t c_frac = c - c_floor;
        const real_t r_frac = r - r_floor;

        const scalar_t p00 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor, c_floor);
        const scalar_t p01 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor, c_floor + 1);
        const scalar_t p10 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor);
        const scalar_t p11 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor + 1);

        const scalar_t p0 = p00 + (p01 - p00) * c_frac;
        const scalar_t p1 = p10 + (p11 - p10) * c_frac;
        return p0 + (p1 - p0) * r_frac;
    }
    
    std::tuple<scalar_t, scalar_t, scalar_t> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const override {
        const int64_t c_floor = floor(c);
        const int64_t r_floor = floor(r);

        const real_t c_frac = c - c_floor;
        const real_t r_frac = r - r_floor;

        const scalar_t p00 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor, c_floor);
        const scalar_t p01 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor, c_floor + 1);
        const scalar_t p10 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor);
        const scalar_t p11 = sample_fftw_with_conjugate(rec, b, boxsize, boxsize_half, r_floor + 1, c_floor + 1);

        // Value computation
        const scalar_t p0 = p00 + (p01 - p00) * c_frac;
        const scalar_t p1 = p10 + (p11 - p10) * c_frac;
        const scalar_t val = p0 + (p1 - p0) * r_frac;

        // Analytical spatial gradients based on Mathematica results
        const scalar_t grad_r = (1 - c_frac) * (p10 - p00) + c_frac * (p11 - p01);
        const scalar_t grad_c = (1 - r_frac) * (p01 - p00) + r_frac * (p11 - p10);

        return std::make_tuple(val, grad_r, grad_c);
    }
};

// Bicubic interpolation kernel (placeholder for future implementation)
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
struct BicubicKernel : public InterpolationKernel<scalar_t, real_t> {
    scalar_t interpolate(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const override {
        // TODO: Implement bicubic interpolation
        // For now, fall back to bilinear
        BilinearKernel<scalar_t, real_t> bilinear;
        return bilinear.interpolate(rec, b, boxsize, boxsize_half, r, c);
    }
    
    std::tuple<scalar_t, scalar_t, scalar_t> interpolate_with_gradients(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
        const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
        real_t r, real_t c
    ) const override {
        // TODO: Implement bicubic gradients
        // For now, fall back to bilinear
        BilinearKernel<scalar_t, real_t> bilinear;
        return bilinear.interpolate_with_gradients(rec, b, boxsize, boxsize_half, r, c);
    }
};

// Factory function to get interpolation kernel
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
std::unique_ptr<InterpolationKernel<scalar_t, real_t>> get_interpolation_kernel(const std::string& interpolation) {
    if (interpolation == "linear") {
        return std::make_unique<BilinearKernel<scalar_t, real_t>>();
    } else if (interpolation == "cubic") {
        return std::make_unique<BicubicKernel<scalar_t, real_t>>();
    } else {
        throw std::runtime_error("Unsupported interpolation method: " + interpolation);
    }
}

// Backward compatibility functions
template <typename scalar_t, typename real_t>
inline scalar_t sample_bilinear_with_conjugate(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
    const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
    real_t r, real_t c
) {
    BilinearKernel<scalar_t, real_t> kernel;
    return kernel.interpolate(rec, b, boxsize, boxsize_half, r, c);
}

template <typename scalar_t, typename real_t>
inline std::tuple<scalar_t, scalar_t, scalar_t> sample_bilinear_with_gradients(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::DefaultPtrTraits>& rec,
    const int64_t b, const int64_t boxsize, const int64_t boxsize_half,
    real_t r, real_t c
) {
    BilinearKernel<scalar_t, real_t> kernel;
    return kernel.interpolate_with_gradients(rec, b, boxsize, boxsize_half, r, c);
}


at::Tensor forward_project_2d_cpu(
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef output_shape,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic", 
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(reconstruction.is_complex(), "Reconstruction must be a complex tensor");
    TORCH_CHECK(reconstruction.dim() == 3, "Reconstruction must be a 3D tensor (B, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2, "Rotations must be (B_rot, P, 2, 2)");

    const auto B = reconstruction.size(0);
    const auto boxsize = reconstruction.size(1);
    const auto boxsize_half = reconstruction.size(2);
    
    const auto B_rot = rotations.size(0);
    const auto P = rotations.size(1);
    TORCH_CHECK(B_rot == B || B_rot == 1, "Batch size of rotations must be 1 or same as reconstruction");

    const auto proj_boxsize = output_shape[0];
    const auto proj_boxsize_half = output_shape[0] / 2 + 1;
    
    auto projection = torch::zeros({B, P, proj_boxsize, proj_boxsize_half}, reconstruction.options());

    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "forward_project_2d_cpu_rotations", ([&] {
        using rot_real_t = scalar_t;
        auto rot_acc = rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> shifts_acc;
        if (shifts.has_value()) {
            TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
            TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1, "Batch size of shifts must be 1 or same as reconstruction");
            TORCH_CHECK(shifts->size(1) == P, "Number of poses in shifts must match rotations");
            TORCH_CHECK(shifts->scalar_type() == rotations.scalar_type(), "Shifts and rotations must have the same dtype");
            shifts_acc.emplace(shifts->packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
        }
        
        AT_DISPATCH_COMPLEX_TYPES(reconstruction.scalar_type(), "forward_project_2d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;
            auto rec_acc = reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();
            auto proj_acc = projection.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();

            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;

            for (int64_t b = 0; b < B; ++b) {
                for (int64_t p = 0; p < P; ++p) {
                    const int64_t rot_b_idx = (B_rot == 1) ? 0 : b;
                    for (int64_t i = 0; i < proj_boxsize; ++i) {
                        for (int64_t j = 0; j < proj_boxsize_half; ++j) {
                            real_t proj_coord_c = j;
                            real_t proj_coord_r = (i <= proj_boxsize / 2) ? i : i - proj_boxsize;

                            if (proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r > radius_cutoff_sq) {
                                continue;
                            }
                            
                            real_t sample_c = proj_coord_c * oversampling;
                            real_t sample_r = proj_coord_r * oversampling;
                            
                            real_t rot_c = rot_acc[rot_b_idx][p][0][0] * sample_c + rot_acc[rot_b_idx][p][0][1] * sample_r;
                            real_t rot_r = rot_acc[rot_b_idx][p][1][0] * sample_c + rot_acc[rot_b_idx][p][1][1] * sample_r;

                            // Use abstracted interpolation kernel
                            auto kernel = get_interpolation_kernel<scalar_t>(interpolation);
                            scalar_t val = kernel->interpolate(rec_acc, b, boxsize, boxsize_half, rot_r, rot_c);
                            
                            if (shifts_acc.has_value()) {
                                const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                                real_t phase = -2.0 * M_PI * (proj_coord_r * (*shifts_acc)[shift_b_idx][p][0] / boxsize + proj_coord_c * (*shifts_acc)[shift_b_idx][p][1] / boxsize);
                                scalar_t phase_factor = scalar_t(cos(phase), sin(phase));
                                val = val * phase_factor;
                            }
                            
                            proj_acc[b][p][i][j] = val;
                        }
                    }
                }
            }
        }));
    }));

    return projection;
}

at::Tensor backward_project_2d_cpu(
    const at::Tensor& grad_projections,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef reconstruction_shape,
    const std::string& interpolation,
    const double oversampling
) {
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic", 
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(grad_projections.is_complex(), "Projections must be a complex tensor");
    TORCH_CHECK(grad_projections.dim() == 4, "Projections must be a 4D tensor (B, P, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2, "Rotations must be (B_rot, P, 2, 2)");

    const auto B = grad_projections.size(0);
    const auto P = grad_projections.size(1);
    const auto proj_boxsize = grad_projections.size(2);
    const auto proj_boxsize_half = grad_projections.size(3);
    
    const auto boxsize = reconstruction_shape[0];
    const auto boxsize_half = reconstruction_shape[1];

    auto grad_reconstruction = torch::zeros({B, boxsize, boxsize_half}, grad_projections.options());

    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "backward_project_2d_cpu_rotations", ([&] {
        using rot_real_t = scalar_t;
        auto rot_acc = rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> shifts_acc;
        if (shifts.has_value()) {
            shifts_acc.emplace(shifts->packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
        }
        
        AT_DISPATCH_COMPLEX_TYPES(grad_projections.scalar_type(), "backward_project_2d_cpu", ([&] {
            using real_t = typename scalar_t::value_type;
            auto grad_rec_acc = grad_reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();
            auto grad_proj_acc = grad_projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();

            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff_sq = default_radius * default_radius;
            
            for (int64_t b = 0; b < B; ++b) {
                auto accumulate_grad = [&](int64_t r, int64_t c, scalar_t grad) {
                    bool needs_conj = false;
                    if (c < 0) { c = -c; r = -r; needs_conj = true; }
                    
                    if (c >= boxsize_half) return;
                    if (r > boxsize / 2 || r < -boxsize / 2 + 1) return;

                    int64_t r_eff = r < 0 ? boxsize + r : r;
                    
                    if (r_eff >= boxsize) return;

                    grad_rec_acc[b][r_eff][c] += needs_conj ? std::conj(grad) : grad;
                };

                for (int64_t p = 0; p < P; ++p) {
                    const int64_t rot_b_idx = (rotations.size(0) == 1) ? 0 : b;
                    for (int64_t i = 0; i < proj_boxsize; ++i) {
                        for (int64_t j = 0; j < proj_boxsize_half; ++j) {
                            real_t proj_coord_c = j;
                            real_t proj_coord_r = (i <= proj_boxsize / 2) ? i : i - proj_boxsize;

                            if (proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r > radius_cutoff_sq) {
                                continue;
                            }

                            real_t sample_c = proj_coord_c * oversampling;
                            real_t sample_r = proj_coord_r * oversampling;

                            real_t rot_c = rot_acc[rot_b_idx][p][0][0] * sample_c + rot_acc[rot_b_idx][p][0][1] * sample_r;
                            real_t rot_r = rot_acc[rot_b_idx][p][1][0] * sample_c + rot_acc[rot_b_idx][p][1][1] * sample_r;

                            scalar_t grad_val = grad_proj_acc[b][p][i][j];
                            
                            if (shifts_acc.has_value()) {
                                const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                                real_t phase = 2.0 * M_PI * (proj_coord_r * (*shifts_acc)[shift_b_idx][p][0] / boxsize + proj_coord_c * (*shifts_acc)[shift_b_idx][p][1] / boxsize);
                                scalar_t phase_factor = scalar_t(cos(phase), sin(phase));
                                grad_val = grad_val * phase_factor;
                            }

                            const int64_t c_floor = floor(rot_c);
                            const int64_t r_floor = floor(rot_r);
                            const real_t c_frac = rot_c - c_floor;
                            const real_t r_frac = rot_r - r_floor;

                            accumulate_grad(r_floor, c_floor, grad_val * (1 - r_frac) * (1 - c_frac));
                            accumulate_grad(r_floor, c_floor + 1, grad_val * (1 - r_frac) * c_frac);
                            accumulate_grad(r_floor + 1, c_floor, grad_val * r_frac * (1 - c_frac));
                            accumulate_grad(r_floor + 1, c_floor + 1, grad_val * r_frac * c_frac);
                        }
                    }
                }
            }
        }));
    }));

    return grad_reconstruction;
} 

std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_project_2d_cpu_adj(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    auto grad_reconstruction = backward_project_2d_cpu(
        grad_projections,
        rotations,
        shifts,
        {reconstruction.size(1), reconstruction.size(2)},
        interpolation,
        oversampling
    );

    const auto B = grad_projections.size(0);
    const auto P = grad_projections.size(1);
    const auto proj_boxsize = grad_projections.size(2);
    const auto proj_boxsize_half = grad_projections.size(3);
    
    const auto rec_boxsize = reconstruction.size(1);
    const auto rec_boxsize_half = reconstruction.size(2);
    
    auto grad_rotations = torch::zeros_like(rotations);
    auto grad_shifts = shifts.has_value() ? torch::zeros_like(*shifts) : torch::empty({0});

    AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "backward_project_2d_cpu_adj_rotations", ([&] {
        using rot_real_t = scalar_t;
        auto rot_acc = rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        auto grad_rot_acc = grad_rotations.packed_accessor32<rot_real_t, 4, torch::DefaultPtrTraits>();
        
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> shifts_acc;
        c10::optional<torch::PackedTensorAccessor32<rot_real_t, 3, torch::DefaultPtrTraits>> grad_shifts_acc;
        if (shifts.has_value()) {
            shifts_acc.emplace(shifts->packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
            grad_shifts_acc.emplace(grad_shifts.packed_accessor32<rot_real_t, 3, torch::DefaultPtrTraits>());
        }

        AT_DISPATCH_COMPLEX_TYPES(grad_projections.scalar_type(), "backward_project_2d_cpu_adj", ([&] {
            using real_t = typename scalar_t::value_type;
            auto grad_proj_acc = grad_projections.packed_accessor32<scalar_t, 4, torch::DefaultPtrTraits>();
            auto rec_acc = reconstruction.packed_accessor32<scalar_t, 3, torch::DefaultPtrTraits>();

            const real_t default_radius = proj_boxsize / 2.0;
            const real_t radius_cutoff = fourier_radius_cutoff.value_or(default_radius);
            const real_t radius_cutoff_sq = radius_cutoff * radius_cutoff;

            for (int64_t b = 0; b < B; ++b) {
                for (int64_t p = 0; p < P; ++p) {
                    const int64_t rot_b_idx = (rotations.size(0) == 1) ? 0 : b;
                    for (int64_t i = 0; i < proj_boxsize; ++i) {
                        for (int64_t j = 0; j < proj_boxsize_half; ++j) {
                            real_t proj_coord_c = j;
                            real_t proj_coord_r = (i <= proj_boxsize / 2) ? i : i - proj_boxsize;

                            if (proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r > radius_cutoff_sq) {
                                continue;
                            }

                            real_t sample_c = proj_coord_c * oversampling;
                            real_t sample_r = proj_coord_r * oversampling;
                            
                            real_t rot_c = rot_acc[rot_b_idx][p][0][0] * sample_c + rot_acc[rot_b_idx][p][0][1] * sample_r;
                            real_t rot_r = rot_acc[rot_b_idx][p][1][0] * sample_c + rot_acc[rot_b_idx][p][1][1] * sample_r;
                            
                            // Use abstracted interpolation kernel
                            auto kernel = get_interpolation_kernel<scalar_t>(interpolation);
                            scalar_t rec_val = kernel->interpolate(rec_acc, b, rec_boxsize, rec_boxsize_half, rot_r, rot_c);
                            scalar_t grad_proj = grad_proj_acc[b][p][i][j];

                            if (shifts_acc.has_value()) {
                               const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                               real_t phase = 2.0 * M_PI * (proj_coord_r * (*shifts_acc)[shift_b_idx][p][0] / rec_boxsize + proj_coord_c * (*shifts_acc)[shift_b_idx][p][1] / rec_boxsize);
                               scalar_t phase_factor = scalar_t(cos(phase), sin(phase));
                               grad_proj = grad_proj * phase_factor;
                            }

                            if (grad_shifts_acc.has_value()) {
                                const int64_t shift_b_idx = (shifts->size(0) == 1) ? 0 : b;
                                
                                // The rec_val needs to be modulated with the phase factor for correct gradients
                                scalar_t modulated_rec_val = rec_val;
                                if (shifts_acc.has_value()) {
                                    real_t phase = -2.0 * M_PI * (proj_coord_r * (*shifts_acc)[shift_b_idx][p][0] / rec_boxsize + proj_coord_c * (*shifts_acc)[shift_b_idx][p][1] / rec_boxsize);
                                    scalar_t phase_factor = scalar_t(cos(phase), sin(phase));
                                    modulated_rec_val = rec_val * phase_factor;
                                }
                                
                                scalar_t phase_grad_r = scalar_t(0, -2.0 * M_PI * proj_coord_r / rec_boxsize) * modulated_rec_val;
                                scalar_t phase_grad_c = scalar_t(0, -2.0 * M_PI * proj_coord_c / rec_boxsize) * modulated_rec_val;
                                
                                // Use the original grad_proj_acc, not the phase-modulated grad_proj
                                scalar_t original_grad_proj = grad_proj_acc[b][p][i][j];
                                (*grad_shifts_acc)[shift_b_idx][p][0] += (original_grad_proj * std::conj(phase_grad_r)).real();
                                (*grad_shifts_acc)[shift_b_idx][p][1] += (original_grad_proj * std::conj(phase_grad_c)).real();
                            }

                            // Compute analytical rotation gradients using chain rule
                            auto kernel_grad = get_interpolation_kernel<scalar_t>(interpolation);
                            auto [rec_val_unused, grad_r, grad_c] = kernel_grad->interpolate_with_gradients(rec_acc, b, rec_boxsize, rec_boxsize_half, rot_r, rot_c);
                            
                            // Apply chain rule: ∂f/∂R[i][j] = (∂f/∂rot_coord) * (∂rot_coord/∂R[i][j])
                            // Based on Mathematica results:
                            // ∂f/∂R[0][0] = (∂f/∂rot_c) * sample_c
                            // ∂f/∂R[0][1] = (∂f/∂rot_c) * sample_r  
                            // ∂f/∂R[1][0] = (∂f/∂rot_r) * sample_c
                            // ∂f/∂R[1][1] = (∂f/∂rot_r) * sample_r
                            grad_rot_acc[rot_b_idx][p][0][0] += (grad_proj * std::conj(grad_c * sample_c)).real();
                            grad_rot_acc[rot_b_idx][p][0][1] += (grad_proj * std::conj(grad_c * sample_r)).real();
                            grad_rot_acc[rot_b_idx][p][1][0] += (grad_proj * std::conj(grad_r * sample_c)).real();
                            grad_rot_acc[rot_b_idx][p][1][1] += (grad_proj * std::conj(grad_r * sample_r)).real();
                        }
                    }
                }
            }
        }));
    }));

    return std::make_tuple(grad_reconstruction, grad_rotations, grad_shifts);
} 