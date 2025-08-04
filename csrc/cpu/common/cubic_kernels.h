/**
 * Cubic Interpolation Kernel Functions
 * 
 * This header provides the mathematical functions for cubic interpolation
 * using Catmull-Rom basis functions. These are shared between 2D (bicubic)
 * and 3D (tricubic) interpolation implementations.
 */

#pragma once

#include <cmath>

namespace torch_projectors {
namespace cpu {
namespace common {

/**
 * Standard bicubic interpolation basis function with a = -0.5 (Catmull-Rom)
 * 
 * This is the classical cubic kernel used in image processing:
 * - Provides C1 continuity (smooth first derivatives)
 * - Support region: [-2, 2]
 * - Interpolates through control points (passes through data exactly)
 */
template <typename real_t>
inline real_t cubic_kernel(real_t s) {
    const real_t a = -0.5;  // Catmull-Rom parameter for optimal smoothness
    s = std::abs(s);  // Kernel is symmetric around 0
    
    if (s <= 1.0) {
        // Inner region: (a+2)|s|³ - (a+3)|s|² + 1
        // This region ensures interpolation (passes through control points)
        return (a + 2.0) * s * s * s - (a + 3.0) * s * s + 1.0;
    } else if (s <= 2.0) {
        // Outer region: a|s|³ - 5a|s|² + 8a|s| - 4a
        // This region provides smooth blending with neighboring samples
        return a * s * s * s - 5.0 * a * s * s + 8.0 * a * s - 4.0 * a;
    } else {
        // Beyond support region: no contribution
        return 0.0;
    }
}

/**
 * Derivative of the bicubic interpolation kernel
 * 
 * Required for computing gradients w.r.t. spatial coordinates.
 * Used in backpropagation for rotation parameter gradients.
 */
template <typename real_t>
inline real_t cubic_kernel_derivative(real_t s) {
    const real_t a = -0.5;
    real_t sign = (s < 0) ? -1.0 : 1.0;  // Preserve sign for derivative
    s = std::abs(s);
    
    if (s <= 1.0) {
        // Inner region derivative: 3(a+2)|s|² - 2(a+3)|s|
        return sign * (3.0 * (a + 2.0) * s * s - 2.0 * (a + 3.0) * s);
    } else if (s <= 2.0) {
        // Outer region derivative: 3a|s|² - 10a|s| + 8a
        return sign * (3.0 * a * s * s - 10.0 * a * s + 8.0 * a);
    } else {
        // Beyond support: no gradient
        return 0.0;
    }
}

} // namespace common
} // namespace cpu
} // namespace torch_projectors