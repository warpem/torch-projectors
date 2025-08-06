// Backprojection-specific utility functions for 3D operations

// Sample gradient from 3D weight reconstruction with bounds and symmetry handling
inline float sample_weight_gradient_3d(
    device const float* grad_weight_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    int32_t d, int32_t r, int32_t c
) {
    // Handle 3D Friedel symmetry and bounds
    if (c < 0) { 
        c = -c;
        r = -r;
        d = -d;
    }
    if (c >= boxsize_half) return 0.0;
    if (d > boxsize / 2 || d < -boxsize / 2 + 1) return 0.0;
    if (r > boxsize / 2 || r < -boxsize / 2 + 1) return 0.0;
    
    // Convert negative indices to positive (FFTW wrapping)
    int32_t d_eff = d < 0 ? boxsize + d : d;
    int32_t r_eff = r < 0 ? boxsize + r : r;
    if (d_eff >= boxsize || r_eff >= boxsize) return 0.0;
    
    return grad_weight_rec[b * rec_batch_stride + d_eff * rec_depth_stride + r_eff * rec_row_stride + c];
}

// Helper function to safely accumulate 3D complex data with Friedel symmetry
inline void accumulate_3d_data_with_symmetry(
    device float2* data_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    int32_t depth, int32_t row, int32_t col, float2 data
) {
    bool needs_conj = false;
    
    // Handle 3D Friedel symmetry for negative column indices
    if (col < 0) { 
        col = -col;
        row = -row;
        depth = -depth;
        needs_conj = true;
    }
    
    // 3D bounds checking
    if (col >= boxsize_half) return;
    if (depth > boxsize / 2 || depth < -boxsize / 2 + 1) return;
    if (row > boxsize / 2 || row < -boxsize / 2 + 1) return;
    
    // Convert negative indices to positive (FFTW wrapping)
    int32_t d_eff = depth < 0 ? boxsize + depth : depth;
    int32_t r_eff = row < 0 ? boxsize + row : row;
    if (d_eff >= boxsize || r_eff >= boxsize) return;
    
    // Calculate linear index and atomically accumulate
    int32_t idx = b * rec_batch_stride + d_eff * rec_depth_stride + r_eff * rec_row_stride + col;
    float2 final_data = needs_conj ? complex_conj(data) : data;
    atomic_add_complex(&data_rec[idx], final_data);

    // On the x=0 line, also insert Friedel-symmetric conjugate counterpart
    if (col == 0) {
        int32_t d_eff2 = (-depth) < 0 ? boxsize + (-depth) : (-depth);
        int32_t r_eff2 = (-row) < 0 ? boxsize + (-row) : (-row);
        
        // Avoid duplicate accumulation at the same position
        if (d_eff2 >= boxsize || r_eff2 >= boxsize || 
            (d_eff2 == d_eff && r_eff2 == r_eff)) return;

        atomic_add_complex(&data_rec[b * rec_batch_stride + d_eff2 * rec_depth_stride + r_eff2 * rec_row_stride + col], 
                          complex_conj(final_data));
    }
}

// Helper function to safely accumulate 3D real weights with Friedel symmetry
inline void accumulate_3d_weights_with_symmetry(
    device float* weight_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    int32_t depth, int32_t row, int32_t col, float weight
) {
    // Handle 3D Friedel symmetry for negative column indices
    if (col < 0) { 
        col = -col;
        row = -row;
        depth = -depth;
    }
    
    // 3D bounds checking
    if (col >= boxsize_half) return;
    if (depth > boxsize / 2 || depth < -boxsize / 2 + 1) return;
    if (row > boxsize / 2 || row < -boxsize / 2 + 1) return;
    
    // Convert negative indices to positive (FFTW wrapping)
    int32_t d_eff = depth < 0 ? boxsize + depth : depth;
    int32_t r_eff = row < 0 ? boxsize + row : row;
    if (d_eff >= boxsize || r_eff >= boxsize) return;
    
    // Calculate linear index and atomically accumulate (weight is always real and positive)
    int32_t idx = b * rec_batch_stride + d_eff * rec_depth_stride + r_eff * rec_row_stride + col;
    atomic_add_real(&weight_rec[idx], weight);

    // On the x=0 line, also insert Friedel-symmetric counterpart
    if (col == 0) {
        int32_t d_eff2 = (-depth) < 0 ? boxsize + (-depth) : (-depth);
        int32_t r_eff2 = (-row) < 0 ? boxsize + (-row) : (-row);
        
        // Avoid duplicate accumulation at the same position
        if (d_eff2 >= boxsize || r_eff2 >= boxsize || 
            (d_eff2 == d_eff && r_eff2 == r_eff)) return;

        atomic_add_real(&weight_rec[b * rec_batch_stride + d_eff2 * rec_depth_stride + r_eff2 * rec_row_stride + col], weight);
    }
}

// Distribute trilinear complex data to 2x2x2 neighborhood
inline void distribute_trilinear_data(
    device float2* data_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    float2 data_val, float d, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    int32_t d_floor = (int32_t)floor(d);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    float d_frac = d - d_floor;
    
    // Distribute data to 2x2x2 neighborhood with trilinear weights
    accumulate_3d_data_with_symmetry(data_rec, b, boxsize, boxsize_half, 
                                    rec_batch_stride, rec_depth_stride, rec_row_stride,
                                    d_floor,     r_floor,     c_floor,     
                                    complex_scale(data_val, (1.0 - d_frac) * (1.0 - r_frac) * (1.0 - c_frac)));
    accumulate_3d_data_with_symmetry(data_rec, b, boxsize, boxsize_half, 
                                    rec_batch_stride, rec_depth_stride, rec_row_stride,
                                    d_floor,     r_floor,     c_floor + 1, 
                                    complex_scale(data_val, (1.0 - d_frac) * (1.0 - r_frac) * c_frac));
    accumulate_3d_data_with_symmetry(data_rec, b, boxsize, boxsize_half, 
                                    rec_batch_stride, rec_depth_stride, rec_row_stride,
                                    d_floor,     r_floor + 1, c_floor,     
                                    complex_scale(data_val, (1.0 - d_frac) * r_frac * (1.0 - c_frac)));
    accumulate_3d_data_with_symmetry(data_rec, b, boxsize, boxsize_half, 
                                    rec_batch_stride, rec_depth_stride, rec_row_stride,
                                    d_floor,     r_floor + 1, c_floor + 1, 
                                    complex_scale(data_val, (1.0 - d_frac) * r_frac * c_frac));
    accumulate_3d_data_with_symmetry(data_rec, b, boxsize, boxsize_half, 
                                    rec_batch_stride, rec_depth_stride, rec_row_stride,
                                    d_floor + 1, r_floor,     c_floor,     
                                    complex_scale(data_val, d_frac * (1.0 - r_frac) * (1.0 - c_frac)));
    accumulate_3d_data_with_symmetry(data_rec, b, boxsize, boxsize_half, 
                                    rec_batch_stride, rec_depth_stride, rec_row_stride,
                                    d_floor + 1, r_floor,     c_floor + 1, 
                                    complex_scale(data_val, d_frac * (1.0 - r_frac) * c_frac));
    accumulate_3d_data_with_symmetry(data_rec, b, boxsize, boxsize_half, 
                                    rec_batch_stride, rec_depth_stride, rec_row_stride,
                                    d_floor + 1, r_floor + 1, c_floor,     
                                    complex_scale(data_val, d_frac * r_frac * (1.0 - c_frac)));
    accumulate_3d_data_with_symmetry(data_rec, b, boxsize, boxsize_half, 
                                    rec_batch_stride, rec_depth_stride, rec_row_stride,
                                    d_floor + 1, r_floor + 1, c_floor + 1, 
                                    complex_scale(data_val, d_frac * r_frac * c_frac));
}

// Distribute tricubic complex data to 4x4x4 neighborhood
inline void distribute_tricubic_data(
    device float2* data_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    float2 data_val, float d, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    int32_t d_floor = (int32_t)floor(d);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    float d_frac = d - d_floor;
    
    // Distribute data to 4x4x4 neighborhood using tricubic weights
    for (int k = -1; k <= 2; ++k) {      // Depth offset
        float weight_d = tricubic_kernel(d_frac - k);
        
        for (int i = -1; i <= 2; ++i) {  // Row offset
            float weight_r = tricubic_kernel(r_frac - i);
            
            for (int j = -1; j <= 2; ++j) {  // Column offset
                float weight_c = tricubic_kernel(c_frac - j);
                float total_weight = weight_d * weight_r * weight_c;
                
                // Only distribute if weight is non-zero
                if (total_weight != 0.0) {
                    accumulate_3d_data_with_symmetry(data_rec, b, boxsize, boxsize_half, 
                                                    rec_batch_stride, rec_depth_stride, rec_row_stride,
                                                    d_floor + k, r_floor + i, c_floor + j, 
                                                    complex_scale(data_val, total_weight));
                }
            }
        }
    }
}

// Distribute trilinear weights to 2x2x2 neighborhood
inline void distribute_trilinear_weights(
    device float* weight_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    float weight_val, float d, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    int32_t d_floor = (int32_t)floor(d);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    float d_frac = d - d_floor;
    
    // Distribute weights to 2x2x2 neighborhood with trilinear weights
    accumulate_3d_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, 
                                       rec_batch_stride, rec_depth_stride, rec_row_stride,
                                       d_floor,     r_floor,     c_floor,     
                                       weight_val * (1.0 - d_frac) * (1.0 - r_frac) * (1.0 - c_frac));
    accumulate_3d_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, 
                                       rec_batch_stride, rec_depth_stride, rec_row_stride,
                                       d_floor,     r_floor,     c_floor + 1, 
                                       weight_val * (1.0 - d_frac) * (1.0 - r_frac) * c_frac);
    accumulate_3d_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, 
                                       rec_batch_stride, rec_depth_stride, rec_row_stride,
                                       d_floor,     r_floor + 1, c_floor,     
                                       weight_val * (1.0 - d_frac) * r_frac * (1.0 - c_frac));
    accumulate_3d_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, 
                                       rec_batch_stride, rec_depth_stride, rec_row_stride,
                                       d_floor,     r_floor + 1, c_floor + 1, 
                                       weight_val * (1.0 - d_frac) * r_frac * c_frac);
    accumulate_3d_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, 
                                       rec_batch_stride, rec_depth_stride, rec_row_stride,
                                       d_floor + 1, r_floor,     c_floor,     
                                       weight_val * d_frac * (1.0 - r_frac) * (1.0 - c_frac));
    accumulate_3d_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, 
                                       rec_batch_stride, rec_depth_stride, rec_row_stride,
                                       d_floor + 1, r_floor,     c_floor + 1, 
                                       weight_val * d_frac * (1.0 - r_frac) * c_frac);
    accumulate_3d_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, 
                                       rec_batch_stride, rec_depth_stride, rec_row_stride,
                                       d_floor + 1, r_floor + 1, c_floor,     
                                       weight_val * d_frac * r_frac * (1.0 - c_frac));
    accumulate_3d_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, 
                                       rec_batch_stride, rec_depth_stride, rec_row_stride,
                                       d_floor + 1, r_floor + 1, c_floor + 1, 
                                       weight_val * d_frac * r_frac * c_frac);
}

// Distribute tricubic weights to 4x4x4 neighborhood
inline void distribute_tricubic_weights(
    device float* weight_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    float weight_val, float d, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    int32_t d_floor = (int32_t)floor(d);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    float d_frac = d - d_floor;
    
    // Distribute weights to 4x4x4 neighborhood using tricubic weights
    for (int k = -1; k <= 2; ++k) {      // Depth offset
        float weight_d = tricubic_kernel(d_frac - k);
        
        for (int i = -1; i <= 2; ++i) {  // Row offset
            float weight_r = tricubic_kernel(r_frac - i);
            
            for (int j = -1; j <= 2; ++j) {  // Column offset
                float weight_c = tricubic_kernel(c_frac - j);
                float total_weight = weight_d * weight_r * weight_c;
                
                // Only distribute if weight is non-zero
                if (total_weight != 0.0) {
                    // For weights, we only need the interpolation weight magnitude (absolute value)
                    accumulate_3d_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, 
                                                       rec_batch_stride, rec_depth_stride, rec_row_stride,
                                                       d_floor + k, r_floor + i, c_floor + j, 
                                                       weight_val * abs(total_weight));
                }
            }
        }
    }
}