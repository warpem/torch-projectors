// Forward backproject 2D kernel - accumulates 2D projections into 2D reconstructions
kernel void backproject_2d_forw_kernel(
    device const float2* projections      [[buffer(0)]],
    device const float*  weights          [[buffer(1)]], 
    device const float*  rotations        [[buffer(2)]],
    device const float*  shifts           [[buffer(3)]],
    device float2*       data_reconstruction [[buffer(4)]],
    device float*        weight_reconstruction [[buffer(5)]],
    constant Params&     params           [[buffer(6)]],
    uint2 gpos [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint2 tpg  [[threads_per_threadgroup]]
) {
    // Thread organization: each threadgroup handles one entire projection
    // Grid: (P, B, 1), Threadgroup: (256, 1, 1)
    int32_t p = gpos.x;  // Pose index
    int32_t b = gpos.y;  // Batch index
    
    // Check bounds for batch and pose
    if (p >= params.P || b >= params.B) {
        return;
    }
    
    // Pre-compute shared parameters for this projection (pose p, batch b)
    
    // Rotation matrix (shared for all pixels in this projection)
    int rot_b_idx = (params.B_rot == 1) ? 0 : b;
    int rot_idx_base = ((rot_b_idx * params.P + p) * 2 * 2);  // Index to start of 2x2 matrix
    float rot_00 = rotations[rot_idx_base + 0];     // R[0][0]
    float rot_01 = rotations[rot_idx_base + 1];     // R[0][1] 
    float rot_10 = rotations[rot_idx_base + 2];     // R[1][0]
    float rot_11 = rotations[rot_idx_base + 3];     // R[1][1]
    
    // Shifts (shared for all pixels in this projection)
    float shift_r = 0.0, shift_c = 0.0;
    if (params.has_shifts) {
        int shift_b_idx = (params.B_shift == 1) ? 0 : b;
        int shift_idx_base = ((shift_b_idx * params.P + p) * 2);
        shift_r = shifts[shift_idx_base + 0];
        shift_c = shifts[shift_idx_base + 1];
    }
    
    // Pre-compute constants
    int32_t proj_batch_stride = params.P * params.proj_boxsize * params.proj_boxsize_half;
    int32_t proj_pose_stride = params.proj_boxsize * params.proj_boxsize_half;
    int32_t proj_base_idx = b * proj_batch_stride + p * proj_pose_stride;
    
    int32_t rec_batch_stride = params.boxsize * params.boxsize_half;
    int32_t rec_row_stride = params.boxsize_half;
    
    float fourier_cutoff_sq = params.fourier_radius_cutoff * params.fourier_radius_cutoff;
    
    bool has_weights = (weights != 0);
    
    // Loop over all pixels in this projection
    int32_t total_pixels = params.proj_boxsize * params.proj_boxsize_half;
    
    for (int32_t pixel_idx = tid; pixel_idx < total_pixels; pixel_idx += tpg.x) {
        // Convert linear pixel index to (i, j) coordinates
        int32_t i = pixel_idx / params.proj_boxsize_half;  // Row
        int32_t j = pixel_idx % params.proj_boxsize_half;  // Column
        
        // Convert array indices to Fourier coordinates (same as CPU)
        float proj_coord_c = float(j);  // Column: always positive (FFTW half-space)
        float proj_coord_r = (i <= params.proj_boxsize / 2) ? float(i) : float(i) - float(params.proj_boxsize);
        
        // Apply Fourier space filtering
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            continue;
        }
        
        // Skip Friedel-symmetric half of the x = 0 line (handled by other half)
        if (j == 0 && i >= params.proj_boxsize / 2) {
            continue;
        }
        
        // Apply oversampling scaling to coordinates
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Apply rotation matrix (using pre-computed matrix elements)
        float rot_c = rot_00 * sample_c + rot_01 * sample_r;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r;
        
        // Get projection data
        float2 proj_val = projections[proj_base_idx + pixel_idx];
        
        // Apply conjugate phase shift for back-projection
        if (params.has_shifts) {
            float phase = -2.0 * M_PI_F * (proj_coord_r * shift_r / params.proj_boxsize + 
                                           proj_coord_c * shift_c / params.proj_boxsize);
            float2 phase_factor = float2(cos(phase), sin(phase));
            proj_val = complex_mul(proj_val, complex_conj(phase_factor));  // conjugate for backprojection
        }
        
        // Distribute projection data to reconstruction 
        if (params.interpolation_method == 0) {  // linear
            distribute_bilinear_data(data_reconstruction, b, params.boxsize, params.boxsize_half,
                                   rec_batch_stride, rec_row_stride, proj_val, rot_r, rot_c);
        } else {  // cubic
            distribute_bicubic_data(data_reconstruction, b, params.boxsize, params.boxsize_half,
                                  rec_batch_stride, rec_row_stride, proj_val, rot_r, rot_c);
        }
        
        // Distribute weights if provided
        if (has_weights) {
            float weight_val = weights[proj_base_idx + pixel_idx];
            
            if (params.interpolation_method == 0) {  // linear
                distribute_bilinear_weights(weight_reconstruction, b, params.boxsize, params.boxsize_half,
                                          rec_batch_stride, rec_row_stride, weight_val, rot_r, rot_c);
            } else {  // cubic  
                distribute_bicubic_weights(weight_reconstruction, b, params.boxsize, params.boxsize_half,
                                         rec_batch_stride, rec_row_stride, weight_val, rot_r, rot_c);
            }
        }
    }
}