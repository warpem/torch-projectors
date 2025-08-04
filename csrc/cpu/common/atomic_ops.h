/**
 * Atomic Operations for Thread-Safe Complex Number Accumulation
 * 
 * This header provides thread-safe atomic operations for accumulating complex
 * and real numbers during parallel projection operations. Since std::atomic<std::complex<T>>
 * is not available, we treat complex numbers as pairs of atomic floats.
 */

#pragma once

#include <atomic>
#include <complex>

namespace torch_projectors {
namespace cpu {
namespace common {

/**
 * Atomic accumulation for complex numbers using separate real/imaginary parts
 * 
 * Since std::atomic<std::complex<T>> is not available, we treat complex numbers
 * as pairs of atomic floats and accumulate real/imaginary parts separately.
 * This avoids race conditions when multiple threads write to the same location.
 */
template <typename scalar_t, typename real_t = typename scalar_t::value_type>
inline void atomic_add_complex(scalar_t* target, const scalar_t& value) {
    // Cast to atomic real types for thread-safe accumulation
    std::atomic<real_t>* real_ptr = reinterpret_cast<std::atomic<real_t>*>(target);
    std::atomic<real_t>* imag_ptr = real_ptr + 1;
    
    // Atomically add real and imaginary parts
    real_ptr->fetch_add(value.real(), std::memory_order_relaxed);
    imag_ptr->fetch_add(value.imag(), std::memory_order_relaxed);
}

/**
 * Atomic accumulation for real numbers
 */
template <typename real_t>
inline void atomic_add_real(real_t* target, const real_t& value) {
    std::atomic<real_t>* atomic_ptr = reinterpret_cast<std::atomic<real_t>*>(target);
    atomic_ptr->fetch_add(value, std::memory_order_relaxed);
}

} // namespace common
} // namespace cpu
} // namespace torch_projectors