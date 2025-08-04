#include <torch/extension.h>
#include "cpu/2d/projection_2d_kernels.h"
#include "cpu/2d/backprojection_2d_kernels.h"
#include "cpu/3d/projection_3d_to_2d_kernels.h"

#ifdef __APPLE__
#include "mps/2d/projection_2d_kernels.h"
#include "mps/3d/projection_3d_to_2d_kernels.h"
#endif

#ifdef USE_CUDA
#include "cuda/2d/projection_2d_kernels.h"
#include "cuda/3d/projection_3d_to_2d_kernels.h"
#endif

TORCH_LIBRARY(torch_projectors, m) {
  // 2D->2D projection operators
  m.def("project_2d_forw(Tensor reconstruction, Tensor rotations, Tensor? shifts, int[] output_shape, str interpolation, float oversampling, float? fourier_radius_cutoff) -> Tensor");
  m.def("project_2d_back(Tensor grad_projections, Tensor reconstruction, Tensor rotations, Tensor? shifts, str interpolation, float oversampling, float? fourier_radius_cutoff) -> (Tensor, Tensor, Tensor)");
  
  // 2D back-projection operators (adjoint/transpose operations)
  m.def("backproject_2d_forw(Tensor projections, Tensor? weights, Tensor rotations, Tensor? shifts, str interpolation, float oversampling, float? fourier_radius_cutoff) -> (Tensor, Tensor)");
  m.def("backproject_2d_back(Tensor grad_data_rec, Tensor? grad_weight_rec, Tensor projections, Tensor? weights, Tensor rotations, Tensor? shifts, str interpolation, float oversampling, float? fourier_radius_cutoff) -> (Tensor, Tensor, Tensor, Tensor)");
  
  // 3D->2D projection operators
  m.def("project_3d_to_2d_forw(Tensor reconstruction, Tensor rotations, Tensor? shifts, int[] output_shape, str interpolation, float oversampling, float? fourier_radius_cutoff) -> Tensor");
  m.def("project_3d_to_2d_back(Tensor grad_projections, Tensor reconstruction, Tensor rotations, Tensor? shifts, str interpolation, float oversampling, float? fourier_radius_cutoff) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torch_projectors, CPU, m) {
  // 2D->2D projection implementations
  m.impl("project_2d_forw", &project_2d_forw_cpu);
  m.impl("project_2d_back", &project_2d_back_cpu);
  
  // 2D back-projection implementations
  m.impl("backproject_2d_forw", &backproject_2d_forw_cpu);
  m.impl("backproject_2d_back", &backproject_2d_back_cpu);
  
  // 3D->2D projection implementations
  m.impl("project_3d_to_2d_forw", &project_3d_to_2d_forw_cpu);
  m.impl("project_3d_to_2d_back", &project_3d_to_2d_back_cpu);
}

#ifdef __APPLE__
TORCH_LIBRARY_IMPL(torch_projectors, MPS, m) {
  // 2D->2D projection implementations
  m.impl("project_2d_forw", &project_2d_forw_mps);
  m.impl("project_2d_back", &project_2d_back_mps);
  
  // 3D->2D projection implementations
  m.impl("project_3d_to_2d_forw", &project_3d_to_2d_forw_mps);
  m.impl("project_3d_to_2d_back", &project_3d_to_2d_back_mps);
}
#endif

#ifdef USE_CUDA
TORCH_LIBRARY_IMPL(torch_projectors, CUDA, m) {
  // 2D->2D projection implementations
  m.impl("project_2d_forw", &project_2d_forw_cuda);
  m.impl("project_2d_back", &project_2d_back_cuda);
  
  // 3D->2D projection implementations
  m.impl("project_3d_to_2d_forw", &project_3d_to_2d_forw_cuda);
  m.impl("project_3d_to_2d_back", &project_3d_to_2d_back_cuda);
}
#endif

// Python module initialization for torch.utils.cpp_extension
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // The TORCH_LIBRARY registration above handles the actual operators
  // This is just needed for the Python extension to initialize properly
} 