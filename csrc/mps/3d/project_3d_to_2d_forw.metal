// Main forward 3D->2D projection compute kernel - OPTIMIZED VERSION
kernel void project_3d_to_2d_forw_kernel(
    device const float2* reconstruction [[buffer(0)]],
    device const float*  rotations      [[buffer(1)]],
    device const float*  shifts         [[buffer(2)]],
    device float2*       projections    [[buffer(3)]],
    constant Params3D&   params         [[buffer(4)]],
    uint2 gpos [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint2 tpg  [[threads_per_threadgroup]]
) {
    // Optimized thread organization: each threadgroup handles one entire projection
    // Grid: (P, B, 1), Threadgroup: (256, 1, 1)
    int32_t p = gpos.x;  // Pose index
    int32_t b = gpos.y;  // Batch index
    
    // Check bounds for batch and pose
    if (p >= params.P || b >= params.B) {
        return;
    }
    
    // Pre-compute shared parameters for this projection (pose p, batch b)
    
    // 3x3 Rotation matrix (shared for all pixels in this projection)
    int rot_b_idx = (params.B_rot == 1) ? 0 : b;
    int rot_idx_base = ((rot_b_idx * params.P + p) * 3 * 3);  // Index to start of 3x3 matrix
    float rot_00 = rotations[rot_idx_base + 0];     // R[0][0]
    float rot_01 = rotations[rot_idx_base + 1];     // R[0][1] 
    float rot_02 = rotations[rot_idx_base + 2];     // R[0][2]
    float rot_10 = rotations[rot_idx_base + 3];     // R[1][0]
    float rot_11 = rotations[rot_idx_base + 4];     // R[1][1]
    float rot_12 = rotations[rot_idx_base + 5];     // R[1][2]
    float rot_20 = rotations[rot_idx_base + 6];     // R[2][0]
    float rot_21 = rotations[rot_idx_base + 7];     // R[2][1]
    float rot_22 = rotations[rot_idx_base + 8];     // R[2][2]
    
    // Shifts (shared for all pixels in this projection)
    float shift_r = 0.0, shift_c = 0.0;
    if (params.has_shifts) {
        int shift_b_idx = (params.B_shift == 1) ? 0 : b;
        int shift_idx_base = ((shift_b_idx * params.P + p) * 2);
        shift_r = shifts[shift_idx_base + 0];
        shift_c = shifts[shift_idx_base + 1];
    }
    
    // Pre-compute constants for this projection
    int32_t proj_batch_stride = params.P * params.proj_boxsize * params.proj_boxsize_half;
    int32_t proj_pose_stride = params.proj_boxsize * params.proj_boxsize_half;
    int32_t proj_row_stride = params.proj_boxsize_half;
    int32_t proj_base_idx = b * proj_batch_stride + p * proj_pose_stride;
    
    // 4D reconstruction strides: [B, D, H, W/2+1]
    int32_t rec_batch_stride = params.boxsize * params.boxsize * params.boxsize_half;  // B stride
    int32_t rec_depth_stride = params.boxsize * params.boxsize_half;                   // D stride  
    int32_t rec_row_stride = params.boxsize_half;                                      // H stride
    
    float fourier_cutoff_sq = params.fourier_radius_cutoff * params.fourier_radius_cutoff;
    
    // Loop over all pixels in this projection, with threads cooperating
    int32_t total_pixels = params.proj_boxsize * params.proj_boxsize_half;
    
    for (int32_t pixel_idx = tid; pixel_idx < total_pixels; pixel_idx += tpg.x) {
        // Convert linear pixel index to (i, j) coordinates
        int32_t i = pixel_idx / params.proj_boxsize_half;  // Row
        int32_t j = pixel_idx % params.proj_boxsize_half;  // Column
        
        // Convert array indices to 2D Fourier coordinates (must match CPU logic exactly)
        float proj_coord_c = float(j);  // Column: always positive (FFTW half-space)
        float proj_coord_r = (i <= params.proj_boxsize / 2) ? float(i) : float(i) - float(params.proj_boxsize); // Row: handle wrap-around
        
        // Apply Fourier space filtering (low-pass)
        float radius_sq = proj_coord_c * proj_coord_c + proj_coord_r * proj_coord_r;
        if (radius_sq > fourier_cutoff_sq) {
            projections[proj_base_idx + i * proj_row_stride + j] = float2(0.0, 0.0);
            continue;
        }
        
        // Apply oversampling scaling to 2D coordinates
        // Oversampling > 1 simulates zero-padding in real space
        float sample_c = proj_coord_c * params.oversampling;
        float sample_r = proj_coord_r * params.oversampling;
        
        // Central slice: extend 2D coordinates to 3D with d=0
        float sample_d = 0.0;  // Central slice through origin
        
        // Apply 3x3 rotation matrix to get sampling coordinates in 3D reconstruction
        // Matrix multiplication: [rot_c; rot_r; rot_d] = R * [sample_c; sample_r; sample_d]
        float rot_c = rot_00 * sample_c + rot_01 * sample_r + rot_02 * sample_d;
        float rot_r = rot_10 * sample_c + rot_11 * sample_r + rot_12 * sample_d;
        float rot_d = rot_20 * sample_c + rot_21 * sample_r + rot_22 * sample_d;
        
        // Interpolate from 4D reconstruction at rotated 3D coordinates
        float2 val;
        if (params.interpolation_method == 0) {  // linear (trilinear)
            val = trilinear_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                       rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c);
        } else {  // cubic (tricubic)
            val = tricubic_interpolate(reconstruction, b, params.boxsize, params.boxsize_half,
                                      rec_batch_stride, rec_depth_stride, rec_row_stride, rot_d, rot_r, rot_c);
        }
        
        // Apply phase shift if translations are provided (using pre-computed shifts)
        // Shift in real space = phase modulation in Fourier space
        if (params.has_shifts) {
            float phase = -2.0 * M_PI_F * (proj_coord_r * shift_r / params.boxsize + 
                                           proj_coord_c * shift_c / params.boxsize);
            float2 phase_factor = float2(cos(phase), sin(phase));
            val = complex_mul(val, phase_factor);
        }
        
        projections[proj_base_idx + pixel_idx] = val;
    }
}