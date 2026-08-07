#pragma once

#include <c10/util/complex.h>

namespace torch_projectors {
namespace cpu {
namespace common {

// Helper to extract the real type from a complex type
template <typename T>
struct real_type { using type = T; };

template <typename T>
struct real_type<c10::complex<T>> { using type = T; };

template <typename T>
using real_type_t = typename real_type<T>::type;

// Atomic add for real types using OpenMP
template <typename T>
inline void atomic_add_real(T* ptr, T value) {
    #pragma omp atomic
    *ptr += value;
}

// Atomic add for complex types using OpenMP
template <typename ComplexT>
inline void atomic_add_complex(ComplexT* ptr, ComplexT value) {
    using RealT = real_type_t<ComplexT>;
    auto* real_ptr = reinterpret_cast<RealT*>(ptr);
    auto* imag_ptr = real_ptr + 1;

    #pragma omp atomic
    *real_ptr += value.real();

    #pragma omp atomic
    *imag_ptr += value.imag();
}

} // namespace common
} // namespace cpu
} // namespace torch_projectors