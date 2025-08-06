// Helper function to safely accumulate gradients with Friedel symmetry
inline void accumulate_gradient_with_symmetry(
    device float2* grad_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    int32_t row, int32_t col, float2 grad
) {
    bool needs_conj = false;
    
    // Handle Friedel symmetry for negative column indices
    if (col < 0) { 
        col = -col;
        row = -row;
        needs_conj = true;
    }
    
    // Bounds checking
    if (col >= boxsize_half) return;
    if (row > boxsize / 2 || row < -boxsize / 2 + 1) return;
    
    // Convert negative row indices to positive (FFTW wrapping)
    int32_t r_eff = row < 0 ? boxsize + row : row;
    if (r_eff >= boxsize) return;
    
    // Calculate linear index and atomically accumulate
    int32_t idx = b * rec_batch_stride + r_eff * rec_row_stride + col;
    float2 final_grad = needs_conj ? complex_conj(grad) : grad;
    atomic_add_complex(&grad_rec[idx], final_grad);
}

// Distribute bilinear gradient to 2x2 neighborhood
inline void distribute_bilinear_gradient(
    device float2* grad_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float2 grad_val, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute gradient to 2x2 neighborhood with bilinear weights
    accumulate_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                    r_floor,     c_floor,     complex_scale(grad_val, (1.0 - r_frac) * (1.0 - c_frac)));
    accumulate_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                    r_floor,     c_floor + 1, complex_scale(grad_val, (1.0 - r_frac) * c_frac));
    accumulate_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                    r_floor + 1, c_floor,     complex_scale(grad_val, r_frac * (1.0 - c_frac)));
    accumulate_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                    r_floor + 1, c_floor + 1, complex_scale(grad_val, r_frac * c_frac));
}

// Distribute bicubic gradient to 4x4 neighborhood
inline void distribute_bicubic_gradient(
    device float2* grad_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float2 grad_val, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute gradient to 4x4 neighborhood using bicubic weights
    for (int i = -1; i <= 2; ++i) {
        float weight_r = bicubic_kernel(r_frac - i);
        
        for (int j = -1; j <= 2; ++j) {
            float weight_c = bicubic_kernel(c_frac - j);
            float total_weight = weight_r * weight_c;
            
            // Only distribute if weight is non-zero
            if (total_weight != 0.0) {
                accumulate_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                                 r_floor + i, c_floor + j, complex_scale(grad_val, total_weight));
            }
        }
    }
}

// Unified backward projection kernel (matches CPU structure)
kernel void backward_project_2d_kernel(
    device const float2* grad_projections [[buffer(0)]],
    device const float2* reconstruction   [[buffer(1)]],
    device const float*  rotations        [[buffer(2)]],
    device const float*  shifts           [[buffer(3)]],
    device float2*       grad_reconstruction [[buffer(4)]],
    device float*        grad_rotations   [[buffer(5)]],
    device float*        grad_shifts      [[buffer(6)]],
    constant Params&     params           [[buffer(7)]],
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
    
    // Flags for what gradients to compute (passed via params)
    bool need_rotation_grads = (params.interpolation_method & 0x10) != 0;  // Use bit flag
    bool need_shift_grads = (params.interpolation_method & 0x20) != 0;     // Use bit flag
    int interpolation_method = params.interpolation_method & 0x0F;         // Lower 4 bits
    
    // Loop over all pixels in this projection
    int32_t total_pixels = params.proj_boxsize * params.proj_boxsize_half;
    
    for (int32_t pixel_idx = tid; pixel_idx < total_pixels; pixel_idx += tpg.x) {
        int32_t i = pixel_idx / params.proj_boxsize_half;
        int32_t j = pixel_idx % params.proj_boxsize_half;
        
        // Convert array indices to Fourier coordinates
        float proj_coord_c = float(j);
        float proj_coord_r = (i <= params.proj_boxsize / 2) ? float(i) : float(i) - float(params.proj_boxsize);

        if (j == 0 && i >= params.proj_boxsize / 2) {
            // Skip Friedel-symmetric half of the x = 0 line (handled by other half)
            continue;
        }
        
        // Apply Fourier space filtering
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            continue;
        }
        
        // Apply oversampling scaling
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Apply rotation matrix
        float rot_c = rot_00 * sample_c + rot_01 * sample_r;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r;
        
        // Get gradient from projection
        float2 grad_proj = grad_projections[proj_base_idx + pixel_idx];
        
        // Apply phase shift correction to gradient if needed (same for both rec and rot gradients)
        float2 grad_proj_for_rec = grad_proj;
        if (params.has_shifts) {
            float phase = 2.0 * M_PI_F * (proj_coord_r * shift_r / params.boxsize + 
                                         proj_coord_c * shift_c / params.boxsize);
            float2 phase_factor = float2(cos(phase), sin(phase));
            grad_proj_for_rec = complex_mul(grad_proj_for_rec, phase_factor);
        }
        
        // 1. ALWAYS compute reconstruction gradients (main scatter operation)
        // Use the phase-corrected gradient
        if (interpolation_method == 0) {  // linear
            distribute_bilinear_gradient(grad_reconstruction, b, params.boxsize, params.boxsize_half,
                                       rec_batch_stride, rec_row_stride, grad_proj_for_rec, rot_r, rot_c);
        } else {  // cubic
            distribute_bicubic_gradient(grad_reconstruction, b, params.boxsize, params.boxsize_half,
                                      rec_batch_stride, rec_row_stride, grad_proj_for_rec, rot_r, rot_c);
        }
        
        // 2. Compute rotation gradients (only if needed)
        if (need_rotation_grads) {
            // Get spatial gradients (rec_val not needed for rotation gradients)
            float2 _unused, grad_r, grad_c;
            if (interpolation_method == 0) {  // linear
                bilinear_interpolate_with_gradients(reconstruction, b, params.boxsize, params.boxsize_half,
                                                  rec_batch_stride, rec_row_stride, rot_r, rot_c,
                                                  &_unused, &grad_r, &grad_c);
            } else {  // cubic
                bicubic_interpolate_with_gradients(reconstruction, b, params.boxsize, params.boxsize_half,
                                                 rec_batch_stride, rec_row_stride, rot_r, rot_c,
                                                 &_unused, &grad_r, &grad_c);
            }
            
            // Use the SAME (phase-correct) gradient for rotation as for reconstruction
            const float2 grad_for_rot = grad_proj_for_rec;
            
            // Chain rule: ∂f/∂R[i][j] = (∂f/∂rot_coord) * (∂rot_coord/∂R[i][j])
            // IMPORTANT: Must match CPU version exactly - conjugate AFTER multiplying by sample
            
            local_rot_grad[tid][0] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_c, sample_c)))).x;  // ∂f/∂R[0][0]
            local_rot_grad[tid][1] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_c, sample_r)))).x;  // ∂f/∂R[0][1]
            local_rot_grad[tid][2] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_r, sample_c)))).x;  // ∂f/∂R[1][0]
            local_rot_grad[tid][3] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_r, sample_r)))).x;  // ∂f/∂R[1][1]
        }
        
        // 3. Compute shift gradients (only if needed)
        if (need_shift_grads) {
            // Get reconstruction value
            float2 rec_val;
            if (interpolation_method == 0) {  // linear
                rec_val = bilinear_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                             rec_batch_stride, rec_row_stride, rot_r, rot_c);
            } else {  // cubic
                rec_val = bicubic_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                            rec_batch_stride, rec_row_stride, rot_r, rot_c);
            }
            
            // Apply phase modulation to reconstruction value
            float phase = -2.0 * M_PI_F * (proj_coord_r * shift_r / params.boxsize + 
                                           proj_coord_c * shift_c / params.boxsize);
            float2 phase_factor = float2(cos(phase), sin(phase));
            float2 modulated_rec_val = complex_mul(rec_val, phase_factor);
            
            // Compute phase derivatives: ∂φ/∂shift = -2π * coordinate / boxsize
            float2 phase_grad_r = complex_mul(float2(0.0, -2.0 * M_PI_F * proj_coord_r / params.boxsize), modulated_rec_val);
            float2 phase_grad_c = complex_mul(float2(0.0, -2.0 * M_PI_F * proj_coord_c / params.boxsize), modulated_rec_val);
            
            // Accumulate shift gradients (taking real part)
            local_shift_grad[tid][0] += (complex_mul(grad_proj, complex_conj(phase_grad_r))).x;
            local_shift_grad[tid][1] += (complex_mul(grad_proj, complex_conj(phase_grad_c))).x;
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