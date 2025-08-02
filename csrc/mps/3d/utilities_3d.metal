// Complex number operations using float2 (real, imag)
inline float2 complex_mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 complex_conj(float2 a) {
    return float2(a.x, -a.y);
}

inline float2 complex_add(float2 a, float2 b) {
    return a + b;
}

inline float2 complex_scale(float2 a, float s) {
    return a * s;
}

// Sample from 3D FFTW-formatted Fourier space with automatic Friedel symmetry handling
// Ported from sample_3d_fftw_with_conjugate in CPU version
inline float2 sample_3d_fftw_with_conjugate(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    int32_t d, int32_t r, int32_t c
) {
    bool need_conjugate = false;
    
    // Handle negative kx via 3D Friedel symmetry (c < 0)
    // For real-valued reconstructions: F(-kx,-ky,-kz) = conj(F(kx,ky,kz))
    if (c < 0) {
        c = -c;          // Mirror to positive kx
        r = -r;          // ky must be mirrored as well
        d = -d;          // kz must be mirrored as well
        need_conjugate = !need_conjugate;
    }
    
    // Clamp coordinates to valid array bounds
    c = min(c, (int32_t)boxsize_half - 1);  // Column: [0, boxsize/2]
    
    // Row and depth: [-boxsize/2+1, boxsize/2]
    r = min((int32_t)boxsize / 2, max(r, -(int32_t)boxsize / 2 + 1));
    d = min((int32_t)boxsize / 2, max(d, -(int32_t)boxsize / 2 + 1));
    
    // Convert negative indices to positive (FFTW wrapping)
    if (r < 0) r = (int32_t)boxsize + r;
    if (d < 0) d = (int32_t)boxsize + d;
    
    // Final bounds check
    r = min(r, (int32_t)boxsize - 1);
    d = min(d, (int32_t)boxsize - 1);
    
    // Calculate linear index
    int32_t idx = b * rec_batch_stride + d * rec_depth_stride + r * rec_row_stride + c;
    float2 value = rec[idx];
    
    // Return conjugated value if we used Friedel symmetry
    return need_conjugate ? complex_conj(value) : value;
}

// Trilinear interpolation kernel
inline float2 trilinear_interpolate(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    float d, float r, float c
) {
    // Extract integer and fractional parts of coordinates
    int32_t d_floor = (int32_t)floor(d);
    int32_t r_floor = (int32_t)floor(r);
    int32_t c_floor = (int32_t)floor(c);
    float d_frac = d - d_floor;  // Fractional part [0,1)
    float r_frac = r - r_floor;  // Fractional part [0,1)
    float c_frac = c - c_floor;  // Fractional part [0,1)

    // Sample 2x2x2 = 8 neighboring voxels
    // Using systematic naming: pDRC where D,R,C ∈ {0,1} indicate the offset
    float2 p000 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor,     c_floor);
    float2 p001 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor,     c_floor + 1);
    float2 p010 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor + 1, c_floor);
    float2 p011 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor + 1, c_floor + 1);
    float2 p100 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor,     c_floor);
    float2 p101 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor,     c_floor + 1);
    float2 p110 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor + 1, c_floor);
    float2 p111 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor + 1, c_floor + 1);

    // Trilinear interpolation: interpolate in each dimension sequentially
    // First, interpolate along the c dimension (4 edge interpolations)
    float2 p00 = complex_add(p000, complex_scale(complex_add(p001, -p000), c_frac));  // Back-bottom edge
    float2 p01 = complex_add(p010, complex_scale(complex_add(p011, -p010), c_frac));  // Back-top edge  
    float2 p10 = complex_add(p100, complex_scale(complex_add(p101, -p100), c_frac));  // Front-bottom edge
    float2 p11 = complex_add(p110, complex_scale(complex_add(p111, -p110), c_frac));  // Front-top edge

    // Second, interpolate along the r dimension (2 face interpolations)
    float2 p0 = complex_add(p00, complex_scale(complex_add(p01, -p00), r_frac));  // Back face
    float2 p1 = complex_add(p10, complex_scale(complex_add(p11, -p10), r_frac));  // Front face

    // Finally, interpolate along the d dimension (final result)
    return complex_add(p0, complex_scale(complex_add(p1, -p0), d_frac));
}

// Tricubic interpolation kernel helper functions
inline float tricubic_kernel(float s) {
    const float a = -0.5;  // Catmull-Rom parameter
    s = abs(s);
    
    if (s <= 1.0) {
        return (a + 2.0) * s * s * s - (a + 3.0) * s * s + 1.0;
    } else if (s <= 2.0) {
        return a * s * s * s - 5.0 * a * s * s + 8.0 * a * s - 4.0 * a;
    } else {
        return 0.0;
    }
}

// Safe 3D sampling with edge clamping for tricubic interpolation
inline float2 sample_3d_with_edge_clamping(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    int32_t d, int32_t r, int32_t c
) {
    // For c: after Friedel symmetry, clamp |c| to valid range [0, boxsize_half-1]
    if (abs(c) >= boxsize_half) {
        c = (c < 0) ? -(boxsize_half - 1) : (boxsize_half - 1);
    }
    
    // For r and d: clamp to valid range [-boxsize/2 + 1, boxsize/2]
    r = max(-boxsize / 2 + 1, min(r, boxsize / 2));
    d = max(-boxsize / 2 + 1, min(d, boxsize / 2));
    
    return sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d, r, c);
}

// Tricubic interpolation kernel
inline float2 tricubic_interpolate(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    float d, float r, float c
) {
    // Extract integer and fractional parts
    int32_t d_floor = (int32_t)floor(d);
    int32_t r_floor = (int32_t)floor(r);
    int32_t c_floor = (int32_t)floor(c);
    float d_frac = d - d_floor;
    float r_frac = r - r_floor;
    float c_frac = c - c_floor;
    
    float2 result = float2(0.0, 0.0);
    
    // Sample 4x4x4 = 64 grid around the interpolation point
    // Grid extends from (d_floor-1, r_floor-1, c_floor-1) to (d_floor+2, r_floor+2, c_floor+2)
    for (int k = -1; k <= 2; ++k) {      // Depth offset: covers 4 depths
        float weight_d = tricubic_kernel(d_frac - k);
        
        for (int i = -1; i <= 2; ++i) {  // Row offset: covers 4 rows
            float weight_r = tricubic_kernel(r_frac - i);

            for (int j = -1; j <= 2; ++j) {  // Column offset: covers 4 columns
                float2 sample = sample_3d_with_edge_clamping(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                                            d_floor + k, r_floor + i, c_floor + j);
                // Compute tricubic weights for this grid position (separable)
                float weight_c = tricubic_kernel(c_frac - j);
                // Accumulate weighted contribution
                result = complex_add(result, complex_scale(sample, weight_d * weight_r * weight_c));
            }
        }
    }
    
    return result;
}

// Trilinear interpolation with gradients (for rotation gradient computation)
inline void trilinear_interpolate_with_gradients(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    float d, float r, float c,
    thread float2* val, thread float2* grad_d, thread float2* grad_r, thread float2* grad_c
) {
    int32_t d_floor = (int32_t)floor(d);
    int32_t r_floor = (int32_t)floor(r);
    int32_t c_floor = (int32_t)floor(c);
    
    float d_frac = d - d_floor;
    float r_frac = r - r_floor;
    float c_frac = c - c_floor;

    // Sample 2x2x2 grid
    float2 p000 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor,     c_floor);
    float2 p001 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor,     c_floor + 1);
    float2 p010 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor + 1, c_floor);
    float2 p011 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor,     r_floor + 1, c_floor + 1);
    float2 p100 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor,     c_floor);
    float2 p101 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor,     c_floor + 1);
    float2 p110 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor + 1, c_floor);
    float2 p111 = sample_3d_fftw_with_conjugate(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride, d_floor + 1, r_floor + 1, c_floor + 1);

    // Value computation (same as interpolate method)
    float2 p00 = complex_add(p000, complex_scale(complex_add(p001, -p000), c_frac));
    float2 p01 = complex_add(p010, complex_scale(complex_add(p011, -p010), c_frac));
    float2 p10 = complex_add(p100, complex_scale(complex_add(p101, -p100), c_frac));
    float2 p11 = complex_add(p110, complex_scale(complex_add(p111, -p110), c_frac));
    
    float2 p0 = complex_add(p00, complex_scale(complex_add(p01, -p00), r_frac));
    float2 p1 = complex_add(p10, complex_scale(complex_add(p11, -p10), r_frac));
    
    *val = complex_add(p0, complex_scale(complex_add(p1, -p0), d_frac));

    // Analytical spatial gradients derived from trilinear formula
    // ∂f/∂d, ∂f/∂r, ∂f/∂c computed analytically for efficiency
    *grad_d = complex_add(p1, -p0);  // ∂f/∂d
    *grad_r = complex_add(complex_scale(complex_add(p01, -p00), 1.0 - d_frac), complex_scale(complex_add(p11, -p10), d_frac));  // ∂f/∂r
    *grad_c = complex_add(complex_add(complex_scale(complex_add(p001, -p000), (1.0 - d_frac) * (1.0 - r_frac)),
                                     complex_scale(complex_add(p011, -p010), (1.0 - d_frac) * r_frac)),
                         complex_add(complex_scale(complex_add(p101, -p100), d_frac * (1.0 - r_frac)),
                                    complex_scale(complex_add(p111, -p110), d_frac * r_frac)));  // ∂f/∂c
}

// Tricubic kernel derivative for gradient computation
inline float tricubic_kernel_derivative(float s) {
    const float a = -0.5;
    float sign = (s < 0) ? -1.0 : 1.0;
    s = abs(s);
    
    if (s <= 1.0) {
        return sign * (3.0 * (a + 2.0) * s * s - 2.0 * (a + 3.0) * s);
    } else if (s <= 2.0) {
        return sign * (3.0 * a * s * s - 10.0 * a * s + 8.0 * a);
    } else {
        return 0.0;
    }
}

// Tricubic interpolation with gradients (for rotation gradient computation)
inline void tricubic_interpolate_with_gradients(
    device const float2* rec,
    int32_t b, int32_t boxsize, int32_t boxsize_half,
    int32_t rec_batch_stride, int32_t rec_depth_stride, int32_t rec_row_stride,
    float d, float r, float c,
    thread float2* val, thread float2* grad_d, thread float2* grad_r, thread float2* grad_c
) {
    int32_t d_floor = (int32_t)floor(d);
    int32_t r_floor = (int32_t)floor(r);
    int32_t c_floor = (int32_t)floor(c);
    
    float d_frac = d - d_floor;
    float r_frac = r - r_floor;
    float c_frac = c - c_floor;
    
    *val = float2(0.0, 0.0);       // Interpolated value
    *grad_d = float2(0.0, 0.0);    // Gradient w.r.t. depth coordinate
    *grad_r = float2(0.0, 0.0);    // Gradient w.r.t. row coordinate
    *grad_c = float2(0.0, 0.0);    // Gradient w.r.t. column coordinate
    
    // Sample 4x4x4 grid and compute value + gradients simultaneously
    for (int k = -1; k <= 2; ++k) {
        float weight_d = tricubic_kernel(d_frac - k);
        float dweight_d = tricubic_kernel_derivative(d_frac - k);
        
        for (int i = -1; i <= 2; ++i) {
            float weight_r = tricubic_kernel(r_frac - i);
            float dweight_r = tricubic_kernel_derivative(r_frac - i);
            
            for (int j = -1; j <= 2; ++j) {
                float2 sample = sample_3d_with_edge_clamping(rec, b, boxsize, boxsize_half, rec_batch_stride, rec_depth_stride, rec_row_stride,
                                                            d_floor + k, r_floor + i, c_floor + j);
                
                // Compute weights and their derivatives for this grid position
                float weight_c = tricubic_kernel(c_frac - j);
                float dweight_c = tricubic_kernel_derivative(c_frac - j);
                
                // Accumulate value and gradients (separable cubic kernels)
                *val = complex_add(*val, complex_scale(sample, weight_d * weight_r * weight_c));
                *grad_d = complex_add(*grad_d, complex_scale(sample, dweight_d * weight_r * weight_c));  // Chain rule: ∂f/∂d
                *grad_r = complex_add(*grad_r, complex_scale(sample, weight_d * dweight_r * weight_c));  // Chain rule: ∂f/∂r
                *grad_c = complex_add(*grad_c, complex_scale(sample, weight_d * weight_r * dweight_c));  // Chain rule: ∂f/∂c
            }
        }
    }
}

// Helper to convert float to uint for atomic operations
inline uint float_as_uint(float f) {
    return as_type<uint>(f);
}

// Helper to convert uint to float for atomic operations  
inline float uint_as_float(uint u) {
    return as_type<float>(u);
}

// Atomic add for complex numbers using compare-and-swap workaround
inline void atomic_add_complex(device float2* target, float2 value) {
    // Metal doesn't support float atomics and can't take address of vector elements
    // Treat the float2 as an array of 2 floats and access them separately
    device float* float_ptr = reinterpret_cast<device float*>(target);
    device _atomic<uint>* real_atomic = reinterpret_cast<device _atomic<uint>*>(float_ptr + 0);
    device _atomic<uint>* imag_atomic = reinterpret_cast<device _atomic<uint>*>(float_ptr + 1);
    
    // Atomic add for real part using CAS loop
    uint expected_real = atomic_load_explicit(real_atomic, memory_order_relaxed);
    uint desired_real;
    do {
        float current_real = uint_as_float(expected_real);
        float new_real = current_real + value.x;
        desired_real = float_as_uint(new_real);
    } while (!atomic_compare_exchange_weak_explicit(real_atomic, &expected_real, desired_real,
                                                   memory_order_relaxed, memory_order_relaxed));
    
    // Atomic add for imaginary part using CAS loop
    uint expected_imag = atomic_load_explicit(imag_atomic, memory_order_relaxed);
    uint desired_imag;
    do {
        float current_imag = uint_as_float(expected_imag);
        float new_imag = current_imag + value.y;
        desired_imag = float_as_uint(new_imag);
    } while (!atomic_compare_exchange_weak_explicit(imag_atomic, &expected_imag, desired_imag,
                                                   memory_order_relaxed, memory_order_relaxed));
}

// Atomic add for real numbers using compare-and-swap workaround
inline void atomic_add_real(device float* target, float value) {
    // Metal doesn't support float atomics, use CAS with integer operations
    device _atomic<uint>* atomic_target = reinterpret_cast<device _atomic<uint>*>(target);
    
    uint expected = atomic_load_explicit(atomic_target, memory_order_relaxed);
    uint desired;
    do {
        float current = uint_as_float(expected);
        float new_value = current + value;
        desired = float_as_uint(new_value);
    } while (!atomic_compare_exchange_weak_explicit(atomic_target, &expected, desired,
                                                   memory_order_relaxed, memory_order_relaxed));
}

struct Params3D {
    int B, P, boxsize, boxsize_half;
    int proj_boxsize, proj_boxsize_half, B_rot, B_shift;
    int has_shifts;
    int interpolation_method;
    float oversampling;
    float fourier_radius_cutoff;
};