// Backprojection-specific utility functions for 2D operations

// Sample gradient from weight reconstruction with bounds and symmetry handling
inline float sample_weight_gradient(
    device const float* grad_weight_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    int32_t r, int32_t c
) {
    // Handle Friedel symmetry and bounds
    if (c < 0) { 
        c = -c;
        r = -r;
    }
    if (c >= boxsize_half) return 0.0;
    if (r > boxsize / 2 || r < -boxsize / 2 + 1) return 0.0;
    
    int32_t r_eff = r < 0 ? boxsize + r : r;
    if (r_eff >= boxsize) return 0.0;
    
    return grad_weight_rec[b * rec_batch_stride + r_eff * rec_row_stride + c];
}

// Helper function to safely accumulate complex data with Friedel symmetry
inline void accumulate_data_with_symmetry(
    device float2* data_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    int32_t row, int32_t col, float2 data
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
    float2 final_data = needs_conj ? complex_conj(data) : data;
    atomic_add_complex(&data_rec[idx], final_data);

    // On the x=0 line, also insert Friedel-symmetric conjugate counterpart
    if (col == 0) {
        int32_t r_eff2 = (-row) < 0 ? boxsize + (-row) : (-row);
        if (r_eff2 >= boxsize || r_eff2 == r_eff) return;

        atomic_add_complex(&data_rec[b * rec_batch_stride + r_eff2 * rec_row_stride + col], complex_conj(final_data));
    }
}

// Helper function to safely accumulate real weights with Friedel symmetry
inline void accumulate_weights_with_symmetry(
    device float* weight_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    int32_t row, int32_t col, float weight
) {
    // Handle Friedel symmetry for negative column indices
    if (col < 0) { 
        col = -col;
        row = -row;
    }
    
    // Bounds checking
    if (col >= boxsize_half) return;
    if (row > boxsize / 2 || row < -boxsize / 2 + 1) return;
    
    // Convert negative row indices to positive (FFTW wrapping)
    int32_t r_eff = row < 0 ? boxsize + row : row;
    if (r_eff >= boxsize) return;
    
    // Calculate linear index and atomically accumulate (weight is always real and positive)
    int32_t idx = b * rec_batch_stride + r_eff * rec_row_stride + col;
    atomic_add_real(&weight_rec[idx], weight);

    // On the x=0 line, also insert Friedel-symmetric counterpart
    if (col == 0) {
        int32_t r_eff2 = (-row) < 0 ? boxsize + (-row) : (-row);
        if (r_eff2 >= boxsize || r_eff2 == r_eff) return;

        atomic_add_real(&weight_rec[b * rec_batch_stride + r_eff2 * rec_row_stride + col], weight);
    }
}

// Distribute bilinear complex data to 2x2 neighborhood
inline void distribute_bilinear_data(
    device float2* data_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float2 data_val, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute data to 2x2 neighborhood with bilinear weights
    accumulate_data_with_symmetry(data_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                r_floor,     c_floor,     complex_scale(data_val, (1.0 - r_frac) * (1.0 - c_frac)));
    accumulate_data_with_symmetry(data_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                r_floor,     c_floor + 1, complex_scale(data_val, (1.0 - r_frac) * c_frac));
    accumulate_data_with_symmetry(data_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                r_floor + 1, c_floor,     complex_scale(data_val, r_frac * (1.0 - c_frac)));
    accumulate_data_with_symmetry(data_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                r_floor + 1, c_floor + 1, complex_scale(data_val, r_frac * c_frac));
}

// Distribute bicubic complex data to 4x4 neighborhood
inline void distribute_bicubic_data(
    device float2* data_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float2 data_val, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute data to 4x4 neighborhood using bicubic weights
    for (int i = -1; i <= 2; ++i) {
        float weight_r = bicubic_kernel(r_frac - i);
        
        for (int j = -1; j <= 2; ++j) {
            float weight_c = bicubic_kernel(c_frac - j);
            float total_weight = weight_r * weight_c;
            
            // Only distribute if weight is non-zero
            if (total_weight != 0.0) {
                accumulate_data_with_symmetry(data_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                            r_floor + i, c_floor + j, complex_scale(data_val, total_weight));
            }
        }
    }
}

// Distribute bilinear weights to 2x2 neighborhood
inline void distribute_bilinear_weights(
    device float* weight_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float weight_val, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute weights to 2x2 neighborhood with bilinear weights
    accumulate_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                   r_floor,     c_floor,     weight_val * (1.0 - r_frac) * (1.0 - c_frac));
    accumulate_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                   r_floor,     c_floor + 1, weight_val * (1.0 - r_frac) * c_frac);
    accumulate_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                   r_floor + 1, c_floor,     weight_val * r_frac * (1.0 - c_frac));
    accumulate_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                   r_floor + 1, c_floor + 1, weight_val * r_frac * c_frac);
}

// Distribute bicubic weights to 4x4 neighborhood
inline void distribute_bicubic_weights(
    device float* weight_rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_row_stride,
    float weight_val, float r, float c
) {
    // Extract integer and fractional parts
    int32_t c_floor = (int32_t)floor(c);
    int32_t r_floor = (int32_t)floor(r);
    float c_frac = c - c_floor;
    float r_frac = r - r_floor;
    
    // Distribute weights to 4x4 neighborhood using bicubic weights
    for (int i = -1; i <= 2; ++i) {
        float weight_r = bicubic_kernel(r_frac - i);
        
        for (int j = -1; j <= 2; ++j) {
            float weight_c = bicubic_kernel(c_frac - j);
            float total_weight = weight_r * weight_c;
            
            // Only distribute if weight is non-zero
            if (total_weight != 0.0) {
                // For weights, we only need the interpolation weight magnitude (absolute value)
                accumulate_weights_with_symmetry(weight_rec, b, boxsize, boxsize_half, rec_batch_stride, rec_row_stride,
                                                r_floor + i, c_floor + j, weight_val * abs(total_weight));
            }
        }
    }
}