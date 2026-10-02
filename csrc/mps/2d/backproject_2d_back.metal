// Backward backproject 2D kernel - computes gradients for backprojection operation
kernel void backproject_2d_back_kernel(
    device const float2* grad_data_rec    [[buffer(0)]],
    device const float*  grad_weight_rec  [[buffer(1)]],
    device const float2* projections      [[buffer(2)]],
    device const float*  weights          [[buffer(3)]],
    device const float*  rotations        [[buffer(4)]],
    device const float*  shifts           [[buffer(5)]],
    device float2*       grad_projections [[buffer(6)]],
    device float*        grad_weights     [[buffer(7)]],
    device float*        grad_rotations   [[buffer(8)]],
    device float*        grad_shifts      [[buffer(9)]],
    constant Params&     params           [[buffer(10)]],
    uint2 gpos [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint2 tpg  [[threads_per_threadgroup]]
) {
    // Thread organization: each threadgroup handles one entire projection
    // Grid: (P, B, 1), Threadgroup: (256, 1, 1)
    int32_t p = gpos.x;  // Pose index
    int32_t b = gpos.y;  // Batch index
    
    if (p >= params.P || b >= params.B) {
        return;
    }
    
    // Local accumulators for gradients that need reduction
    threadgroup float local_rot_grad[256][4];    // [thread][matrix_element] 
    threadgroup float local_shift_grad[256][2];  // [thread][shift_component]
    
    // Initialize local accumulators
    local_rot_grad[tid][0] = 0.0;   // R[0][0]
    local_rot_grad[tid][1] = 0.0;   // R[0][1] 
    local_rot_grad[tid][2] = 0.0;   // R[1][0]
    local_rot_grad[tid][3] = 0.0;   // R[1][1]
    local_shift_grad[tid][0] = 0.0; // shift_r
    local_shift_grad[tid][1] = 0.0; // shift_c
    
    // Pre-compute shared parameters for this projection
    int rot_b_idx = (params.B_rot == 1) ? 0 : b;
    int rot_idx_base = ((rot_b_idx * params.P + p) * 2 * 2);
    float rot_00 = rotations[rot_idx_base + 0];
    float rot_01 = rotations[rot_idx_base + 1];
    float rot_10 = rotations[rot_idx_base + 2];
    float rot_11 = rotations[rot_idx_base + 3];
    
    float shift_r = 0.0, shift_c = 0.0;
    if (params.has_shifts) {
        int shift_b_idx = (params.B_shift == 1) ? 0 : b;
        int shift_idx_base = ((shift_b_idx * params.P + p) * 2);
        shift_r = shifts[shift_idx_base + 0];
        shift_c = shifts[shift_idx_base + 1];
    }
    
    // Pre-compute strides
    int32_t proj_batch_stride = params.P * params.proj_boxsize * params.proj_boxsize_half;
    int32_t proj_pose_stride = params.proj_boxsize * params.proj_boxsize_half;
    int32_t proj_base_idx = b * proj_batch_stride + p * proj_pose_stride;
    
    int32_t rec_batch_stride = params.boxsize * params.boxsize_half;
    int32_t rec_row_stride = params.boxsize_half;
    
    float fourier_cutoff_sq = params.fourier_radius_cutoff * params.fourier_radius_cutoff;
    
    // Gradient computation flags (packed in interpolation_method)
    bool need_rotation_grads = (params.interpolation_method & 0x10) != 0;
    bool need_shift_grads = (params.interpolation_method & 0x20) != 0;
    // Optional buffers are represented by explicit flags, not pointer values.
    bool has_weights = (params.interpolation_method & 0x40) != 0;
    bool need_weight_grads = (params.interpolation_method & 0x80) != 0;
    int interpolation_method = params.interpolation_method & 0x0F;
    
    // Loop over all pixels in this projection
    int32_t total_pixels = params.proj_boxsize * params.proj_boxsize_half;
    
    for (int32_t pixel_idx = tid; pixel_idx < total_pixels; pixel_idx += tpg.x) {
        int32_t i = pixel_idx / params.proj_boxsize_half;
        int32_t j = pixel_idx % params.proj_boxsize_half;
        
        // Convert array indices to Fourier coordinates
        float proj_coord_c = float(j);
        float proj_coord_r = (i <= params.proj_boxsize / 2) ? float(i) : float(i) - float(params.proj_boxsize);
        
        // Apply Fourier space filtering
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            continue;
        }
        
        if (j == 0 && i >= params.proj_boxsize / 2) {
            // Skip Friedel-symmetric half of the x = 0 line
            continue;
        }
        
        // Apply oversampling scaling
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Apply rotation matrix
        float rot_c = rot_00 * sample_c + rot_01 * sample_r;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r;
        
        // 1. Compute grad_projections using forward projection (adjoint relationship)
        float2 rec_val;
        if (interpolation_method == 0) {  // linear
            rec_val = bilinear_interpolate(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                         rec_batch_stride, rec_row_stride, rot_r, rot_c, true);
        } else {  // cubic
            rec_val = bicubic_interpolate(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                        rec_batch_stride, rec_row_stride, rot_r, rot_c, true);
        }
        
        // Apply conjugate phase shift (opposite of back-projection)
        if (params.has_shifts) {
            float phase = -2.0 * M_PI_F * (proj_coord_r * shift_r / params.proj_boxsize + 
                                           proj_coord_c * shift_c / params.proj_boxsize);
            float2 phase_factor = float2(cos(phase), sin(phase));
            rec_val = complex_mul(rec_val, phase_factor);  // forward phase for grad_projections
        }
        
        grad_projections[proj_base_idx + pixel_idx] = rec_val;
        
        // Differentiate the positive weight stencil used by insertion, including its pose dependence.
        float weight_coordinate_grad[2] = {};
        if (need_weight_grads && has_weights) {
            const bool weight_cubic = interpolation_method != 0;
            const int width = weight_cubic ? 4 : 2;
            const float coordinates[2] = {rot_c, rot_r};
            int base[2];
            float factors[2][4], derivatives[2][4];
            int tap_count = 1;
            for (int axis = 0; axis < 2; ++axis) {
                const int lower = (int)floor(coordinates[axis]);
                base[axis] = lower - (weight_cubic ? 1 : 0);
                const float fraction = coordinates[axis] - lower;
                for (int tap = 0; tap < width; ++tap) {
                    const float delta = coordinates[axis] - (base[axis] + tap);
                    factors[axis][tap] = weight_cubic ? bicubic_kernel(delta) : (tap ? fraction : 1 - fraction);
                    derivatives[axis][tap] = weight_cubic ? bicubic_kernel_derivative(delta) : (tap ? 1 : -1);
                }
                tap_count *= width;
            }
            float weight_grad = 0;
            for (int flat = 0; flat < tap_count; ++flat) {
                int remaining = flat;
                int tap[2], point[2];
                float coefficient = 1;
                for (int axis = 0; axis < 2; ++axis) {
                    tap[axis] = remaining % width; remaining /= width;
                    point[axis] = base[axis] + tap[axis];
                    coefficient *= factors[axis][tap[axis]];
                }
                const float sampled = sample_weight_gradient(grad_weight_rec, b, params.boxsize, params.boxsize_half, rec_batch_stride, rec_row_stride, point[1], point[0]);
                weight_grad += sampled * abs(coefficient);
                if (need_rotation_grads) {
                    // Cubic insertion uses abs(K); linear coefficients are nonnegative.
                    const float sign = weight_cubic ? (coefficient > 0 ? 1 : (coefficient < 0 ? -1 : 0)) : 1;
                    for (int axis = 0; axis < 2; ++axis) {
                        float derivative = sign;
                        for (int other = 0; other < 2; ++other)
                            derivative *= other == axis ? derivatives[other][tap[other]] : factors[other][tap[other]];
                        weight_coordinate_grad[axis] += sampled * derivative;
                    }
                }
            }
            grad_weights[proj_base_idx + pixel_idx] = weight_grad;
            for (int axis = 0; axis < 2; ++axis)
                weight_coordinate_grad[axis] *= weights[proj_base_idx + pixel_idx];
        }

        // 3. Compute rotation gradients if needed
        if (need_rotation_grads) {
            local_rot_grad[tid][0] += weight_coordinate_grad[0] * sample_c;
            local_rot_grad[tid][1] += weight_coordinate_grad[0] * sample_r;
            local_rot_grad[tid][2] += weight_coordinate_grad[1] * sample_c;
            local_rot_grad[tid][3] += weight_coordinate_grad[1] * sample_r;
            float2 _unused, grad_r, grad_c;
            if (interpolation_method == 0) {  // linear
                bilinear_interpolate_with_gradients(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                                  rec_batch_stride, rec_row_stride, rot_r, rot_c,
                                                  &_unused, &grad_r, &grad_c, true);
            } else {  // cubic
                bicubic_interpolate_with_gradients(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                                 rec_batch_stride, rec_row_stride, rot_r, rot_c,
                                                 &_unused, &grad_r, &grad_c, true);
            }
            
            float2 proj_val = projections[proj_base_idx + pixel_idx];
            
            // Apply conjugate phase shift to projection value
            if (params.has_shifts) {
                float phase = -2.0 * M_PI_F * (proj_coord_r * shift_r / params.proj_boxsize + 
                                               proj_coord_c * shift_c / params.proj_boxsize);
                float2 phase_factor = float2(cos(phase), sin(phase));
                proj_val = complex_mul(proj_val, complex_conj(phase_factor));
            }
            
            // Chain rule for rotation matrix gradients
            local_rot_grad[tid][0] += (complex_mul(proj_val, complex_conj(complex_scale(grad_c, sample_c)))).x;
            local_rot_grad[tid][1] += (complex_mul(proj_val, complex_conj(complex_scale(grad_c, sample_r)))).x;
            local_rot_grad[tid][2] += (complex_mul(proj_val, complex_conj(complex_scale(grad_r, sample_c)))).x;
            local_rot_grad[tid][3] += (complex_mul(proj_val, complex_conj(complex_scale(grad_r, sample_r)))).x;
        }
        
        // 4. Compute shift gradients if needed
        if (need_shift_grads) {
            float2 rec_val_for_shift;
            if (interpolation_method == 0) {  // linear
                rec_val_for_shift = bilinear_interpolate(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                                       rec_batch_stride, rec_row_stride, rot_r, rot_c, true);
            } else {  // cubic
                rec_val_for_shift = bicubic_interpolate(grad_data_rec, b, params.boxsize, params.boxsize_half,
                                                      rec_batch_stride, rec_row_stride, rot_r, rot_c, true);
            }
            
            float2 proj_val = projections[proj_base_idx + pixel_idx];
            
            // Apply conjugate phase shift to projection value (consistent with rotation gradient computation)
            if (params.has_shifts) {
                float phase = -2.0 * M_PI_F * (proj_coord_r * shift_r / params.proj_boxsize + 
                                               proj_coord_c * shift_c / params.proj_boxsize);
                float2 phase_factor = float2(cos(phase), sin(phase));
                proj_val = complex_mul(proj_val, complex_conj(phase_factor));
            }
            
            float2 phase_grad_r = complex_mul(float2(0.0, -2.0 * M_PI_F * proj_coord_r / params.proj_boxsize), rec_val_for_shift);
            float2 phase_grad_c = complex_mul(float2(0.0, -2.0 * M_PI_F * proj_coord_c / params.proj_boxsize), rec_val_for_shift);
            
            local_shift_grad[tid][0] += (complex_mul(proj_val, complex_conj(phase_grad_r))).x;
            local_shift_grad[tid][1] += (complex_mul(proj_val, complex_conj(phase_grad_c))).x;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce and write gradients (only thread 0 writes to avoid races)
    if (tid == 0) {
        // Reduce rotation gradients across threads
        if (need_rotation_grads) {
            float total_rot_grad[4] = {0.0, 0.0, 0.0, 0.0};
            for (uint t = 0; t < tpg.x; ++t) {
                total_rot_grad[0] += local_rot_grad[t][0];
                total_rot_grad[1] += local_rot_grad[t][1];
                total_rot_grad[2] += local_rot_grad[t][2];
                total_rot_grad[3] += local_rot_grad[t][3];
            }
            
            // Atomically add to global rotation gradients
            int global_rot_idx_base = ((rot_b_idx * params.P + p) * 2 * 2);
            atomic_add_real(&grad_rotations[global_rot_idx_base + 0], total_rot_grad[0]);
            atomic_add_real(&grad_rotations[global_rot_idx_base + 1], total_rot_grad[1]);
            atomic_add_real(&grad_rotations[global_rot_idx_base + 2], total_rot_grad[2]);
            atomic_add_real(&grad_rotations[global_rot_idx_base + 3], total_rot_grad[3]);
        }
        
        // Reduce shift gradients across threads
        if (need_shift_grads) {
            float total_shift_grad[2] = {0.0, 0.0};
            for (uint t = 0; t < tpg.x; ++t) {
                total_shift_grad[0] += local_shift_grad[t][0];
                total_shift_grad[1] += local_shift_grad[t][1];
            }
            
            // Atomically add to global shift gradients
            int shift_b_idx = (params.B_shift == 1) ? 0 : b;
            int global_shift_idx_base = ((shift_b_idx * params.P + p) * 2);
            atomic_add_real(&grad_shifts[global_shift_idx_base + 0], total_shift_grad[0]);
            atomic_add_real(&grad_shifts[global_shift_idx_base + 1], total_shift_grad[1]);
        }
    }
}