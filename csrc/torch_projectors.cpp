#include <torch/extension.h>
#include "cpu/cpu_kernels.h"

#ifdef __APPLE__
#include "mps/mps_kernels.h"
#endif

TORCH_LIBRARY(torch_projectors, m) {
  m.def("forward_project_2d(Tensor reconstruction, Tensor rotations, Tensor? shifts, int[] output_shape, str interpolation, float oversampling, float? fourier_radius_cutoff) -> Tensor");
  m.def("backward_project_2d(Tensor grad_projections, Tensor reconstruction, Tensor rotations, Tensor? shifts, str interpolation, float oversampling, float? fourier_radius_cutoff) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torch_projectors, CPU, m) {
  m.impl("forward_project_2d", &forward_project_2d_cpu);
  m.impl("backward_project_2d", &backward_project_2d_cpu);
}

#ifdef __APPLE__
TORCH_LIBRARY_IMPL(torch_projectors, MPS, m) {
  m.impl("forward_project_2d", &forward_project_2d_mps);
  m.impl("backward_project_2d", &backward_project_2d_mps);
}
#endif 