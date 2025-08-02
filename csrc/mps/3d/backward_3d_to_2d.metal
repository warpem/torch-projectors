// Helper function to safely accumulate gradients with 3D Friedel symmetry
inline void accumulate_3d_gradient_with_symmetry(
    device float2* grad_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    int32_t depth, int32_t row, int32_t col, float2 grad
) {
    bool needs_conj = false;
    
    // Handle 3D Friedel symmetry for negative column indices
    if (col < 0) { 
        col = -col;           // Mirror column to positive side
        row = -row;           // Mirror row as well
        depth = -depth;       // Mirror depth as well
        needs_conj = true;    // Need to conjugate the value
    }
    
    // Bounds checking
    if (col >= boxsize_half) return;  // Beyond stored frequency range
    if (row > boxsize / 2 || row < -boxsize / 2 + 1) return;  // Beyond valid row range
    if (depth > boxsize / 2 || depth < -boxsize / 2 + 1) return;  // Beyond valid depth range

    // Convert negative indices to positive (FFTW wrapping)
    int32_t r_eff = row < 0 ? boxsize + row : row;
    int32_t d_eff = depth < 0 ? boxsize + depth : depth;
    if (r_eff >= boxsize || d_eff >= boxsize) return;  // Final bounds check

    // Calculate linear index and atomically accumulate
    int32_t idx = b * rec_batch_stride + d_eff * rec_depth_stride + r_eff * rec_row_stride + col;
    float2 final_grad = needs_conj ? complex_conj(grad) : grad;
    atomic_add_complex(&grad_rec[idx], final_grad);
}

// Distribute trilinear gradient to 2x2x2 neighborhood
inline void distribute_trilinear_gradient(
    device float2* grad_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    float2 grad_val, float d, float r, float c
) {
    // Extract integer and fractional parts
    int32_t d_floor = (int32_t)floor(d);
    int32_t r_floor = (int32_t)floor(r);
    int32_t c_floor = (int32_t)floor(c);
    float d_frac = d - d_floor;
    float r_frac = r - r_floor;
    float c_frac = c - c_floor;
    
    // Distribute gradient to 2x2x2 neighborhood with trilinear weights
    // These are exactly the same weights used in forward trilinear interpolation
    accumulate_3d_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                        d_floor,     r_floor,     c_floor,     complex_scale(grad_val, (1.0 - d_frac) * (1.0 - r_frac) * (1.0 - c_frac))); // p000
    accumulate_3d_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                        d_floor,     r_floor,     c_floor + 1, complex_scale(grad_val, (1.0 - d_frac) * (1.0 - r_frac) * c_frac));       // p001
    accumulate_3d_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                        d_floor,     r_floor + 1, c_floor,     complex_scale(grad_val, (1.0 - d_frac) * r_frac * (1.0 - c_frac)));       // p010
    accumulate_3d_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                        d_floor,     r_floor + 1, c_floor + 1, complex_scale(grad_val, (1.0 - d_frac) * r_frac * c_frac));             // p011
    accumulate_3d_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                        d_floor + 1, r_floor,     c_floor,     complex_scale(grad_val, d_frac * (1.0 - r_frac) * (1.0 - c_frac)));       // p100
    accumulate_3d_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                        d_floor + 1, r_floor,     c_floor + 1, complex_scale(grad_val, d_frac * (1.0 - r_frac) * c_frac));             // p101
    accumulate_3d_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                        d_floor + 1, r_floor + 1, c_floor,     complex_scale(grad_val, d_frac * r_frac * (1.0 - c_frac)));             // p110
    accumulate_3d_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                        d_floor + 1, r_floor + 1, c_floor + 1, complex_scale(grad_val, d_frac * r_frac * c_frac));                   // p111
}

// Distribute tricubic gradient to 4x4x4 neighborhood
inline void distribute_tricubic_gradient(
    device float2* grad_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    float2 grad_val, float d, float r, float c
) {
    // Extract integer and fractional parts
    int32_t d_floor = (int32_t)floor(d);
    int32_t r_floor = (int32_t)floor(r);
    int32_t c_floor = (int32_t)floor(c);
    float d_frac = d - d_floor;
    float r_frac = r - r_floor;
    float c_frac = c - c_floor;
    
    // Distribute gradient to 4x4x4 neighborhood using tricubic weights
    // These are exactly the same weights used in forward tricubic interpolation
    for (int k = -1; k <= 2; ++k) {      // Depth offset: covers 4 depths
        float weight_d = tricubic_kernel(d_frac - k);

        for (int i = -1; i <= 2; ++i) {  // Row offset: covers 4 rows
            float weight_r = tricubic_kernel(r_frac - i);

            for (int j = -1; j <= 2; ++j) {  // Column offset: covers 4 columns
                float weight_c = tricubic_kernel(c_frac - j);
                float total_weight = weight_d * weight_r * weight_c;
                
                // Only distribute if weight is non-zero (tricubic has finite support)
                if (total_weight != 0.0) {
                    accumulate_3d_gradient_with_symmetry(grad_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                                        d_floor + k, r_floor + i, c_floor + j, complex_scale(grad_val, total_weight));
                }
            }
        }
    }
}

// Unified backward 3D->2D projection kernel (matches CPU structure)
kernel void backward_project_3d_to_2d_kernel(
    device const float2* grad_projections [[buffer(0)]],
    device const float2* reconstruction   [[buffer(1)]],
    device const float*  rotations        [[buffer(2)]],
    device const float*  shifts           [[buffer(3)]],
    device float2*       grad_reconstruction [[buffer(4)]],
    device float*        grad_rotations   [[buffer(5)]],
    device float*        grad_shifts      [[buffer(6)]],
    constant Params3D&   params           [[buffer(7)]],
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
    threadgroup float local_rot_grad[256][9];  // [thread][matrix_element] - 3x3 = 9 elements
    threadgroup float local_shift_grad[256][2];  // [thread][shift_component]
    
    // Initialize local accumulators
    local_rot_grad[tid][0] = 0.0;   // R[0][0]
    local_rot_grad[tid][1] = 0.0;   // R[0][1] 
    local_rot_grad[tid][2] = 0.0;   // R[0][2]
    local_rot_grad[tid][3] = 0.0;   // R[1][0]
    local_rot_grad[tid][4] = 0.0;   // R[1][1]
    local_rot_grad[tid][5] = 0.0;   // R[1][2]
    local_rot_grad[tid][6] = 0.0;   // R[2][0]
    local_rot_grad[tid][7] = 0.0;   // R[2][1]
    local_rot_grad[tid][8] = 0.0;   // R[2][2]
    local_shift_grad[tid][0] = 0.0; // shift_r
    local_shift_grad[tid][1] = 0.0; // shift_c
    
    // Pre-compute shared parameters for this projection
    int rot_b_idx = (params.B_rot == 1) ? 0 : b;
    int rot_idx_base = ((rot_b_idx * params.P + p) * 3 * 3);  // 3x3 matrix
    float rot_00 = rotations[rot_idx_base + 0];
    float rot_01 = rotations[rot_idx_base + 1];
    float rot_02 = rotations[rot_idx_base + 2];
    float rot_10 = rotations[rot_idx_base + 3];
    float rot_11 = rotations[rot_idx_base + 4];
    float rot_12 = rotations[rot_idx_base + 5];
    float rot_20 = rotations[rot_idx_base + 6];
    float rot_21 = rotations[rot_idx_base + 7];
    float rot_22 = rotations[rot_idx_base + 8];
    
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
    
    // 4D reconstruction strides: [B, D, H, W/2+1]
    int32_t rec_batch_stride = params.boxsize * params.boxsize * params.boxsize_half;
    int32_t rec_depth_stride = params.boxsize * params.boxsize_half;
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
        
        // Apply Fourier space filtering
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            continue;
        }
        
        // Apply oversampling scaling to 2D coordinates
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Central slice: extend 2D coordinates to 3D with d=0
        float sample_d = 0.0;  // Central slice
        
        // Apply 3x3 rotation matrix to get sampling coordinates in 3D reconstruction
        float rot_c = rot_00 * sample_c + rot_01 * sample_r + rot_02 * sample_d;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r + rot_12 * sample_d;
        float rot_d = rot_20 * sample_c + rot_21 * sample_r + rot_22 * sample_d;
        
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
        
        // 1. ALWAYS compute reconstruction gradients (main scatter operation from 2D to 4D)
        // Use the phase-corrected gradient
        if (interpolation_method == 0) {  // linear (trilinear)
            distribute_trilinear_gradient(grad_reconstruction, b, params.boxsize, params.boxsize_half,
                                        rec_batch_stride, rec_depth_stride, rec_row_stride, grad_proj_for_rec, rot_d, rot_r, rot_c);
        } else {  // cubic (tricubic)
            distribute_tricubic_gradient(grad_reconstruction, b, params.boxsize, params.boxsize_half,
                                       rec_batch_stride, rec_depth_stride, rec_row_stride, grad_proj_for_rec, rot_d, rot_r, rot_c);
        }
        
        // 2. Compute rotation gradients (only if needed)
        if (need_rotation_grads) {
            // Get spatial gradients (rec_val not needed for rotation gradients)
            float2 _unused, grad_d, grad_r, grad_c;
            if (interpolation_method == 0) {  // linear (trilinear)
                trilinear_interpolate_with_gradients(reconstruction, b, params.boxsize, params.boxsize_half,
                                                   rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c,
                                                   &_unused, &grad_d, &grad_r, &grad_c);
            } else {  // cubic (tricubic)
                tricubic_interpolate_with_gradients(reconstruction, b, params.boxsize, params.boxsize_half,
                                                  rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c,
                                                  &_unused, &grad_d, &grad_r, &grad_c);
            }
            
            // Use the SAME (phase-correct) gradient for rotation as for reconstruction
            const float2 grad_for_rot = grad_proj_for_rec;
            
            // Chain rule: ∂f/∂R[i][j] = (∂f/∂rot_coord) * (∂rot_coord/∂R[i][j])
            // 3D Rotation transformation: [rot_c; rot_r; rot_d] = R * [sample_c; sample_r; sample_d]
            // Therefore:
            // ∂rot_c/∂R[0][0] = sample_c, ∂rot_c/∂R[0][1] = sample_r, ∂rot_c/∂R[0][2] = sample_d
            // ∂rot_r/∂R[1][0] = sample_c, ∂rot_r/∂R[1][1] = sample_r, ∂rot_r/∂R[1][2] = sample_d
            // ∂rot_d/∂R[2][0] = sample_c, ∂rot_d/∂R[2][1] = sample_r, ∂rot_d/∂R[2][2] = sample_d
            //
            // Accumulate gradients locally (taking real part for real-valued rotation matrices):
            local_rot_grad[tid][0] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_c, sample_c)))).x;  // ∂f/∂R[0][0]
            local_rot_grad[tid][1] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_c, sample_r)))).x;  // ∂f/∂R[0][1]
            local_rot_grad[tid][2] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_c, sample_d)))).x;  // ∂f/∂R[0][2]
            local_rot_grad[tid][3] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_r, sample_c)))).x;  // ∂f/∂R[1][0]
            local_rot_grad[tid][4] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_r, sample_r)))).x;  // ∂f/∂R[1][1]
            local_rot_grad[tid][5] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_r, sample_d)))).x;  // ∂f/∂R[1][2]
            local_rot_grad[tid][6] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_d, sample_c)))).x;  // ∂f/∂R[2][0]
            local_rot_grad[tid][7] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_d, sample_r)))).x;  // ∂f/∂R[2][1]
            local_rot_grad[tid][8] += (complex_mul(grad_for_rot, complex_conj(complex_scale(grad_d, sample_d)))).x;  // ∂f/∂R[2][2]
        }
        
        // 3. Compute shift gradients (only if needed)
        if (need_shift_grads) {
            // Get reconstruction value
            float2 rec_val;
            if (interpolation_method == 0) {  // linear (trilinear)
                rec_val = trilinear_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                              rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c);
            } else {  // cubic (tricubic)
                rec_val = tricubic_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                             rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c);
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
            float total_rot_grad[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (uint t = 0; t < tpg.x; ++t) {
                for (int i = 0; i < 9; ++i) {
                    total_rot_grad[i] += local_rot_grad[t][i];
                }
            }
            
            // Atomically add to global rotation gradients
            int global_rot_idx_base = ((rot_b_idx * params.P + p) * 3 * 3);
            for (int i = 0; i < 9; ++i) {
                atomic_add_real(&grad_rotations[global_rot_idx_base + i], total_rot_grad[i]);
            }
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