#include "mps_kernels.h"

#ifdef __APPLE__

#include <torch/extension.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <ATen/mps/MPSStream.h>
#include "projection_kernels.h"

// Get Metal compute pipeline for forward projection
id<MTLComputePipelineState> get_forward_projection_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipeline = nil;
    static id<MTLDevice> cached_device = nil;
    static dispatch_once_t onceToken;
    static NSString* shaderSource = nil;
    
    if (!shaderSource) {
        shaderSource = [NSString stringWithUTF8String:PROJECTION_KERNEL_SOURCE];
    }
    
    if (pipeline && device == cached_device) {
        return pipeline;
    }
    
    @autoreleasepool {
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        TORCH_CHECK(library != nil, "Failed to compile Metal library: ", error.localizedDescription.UTF8String);
        
        id<MTLFunction> function = [library newFunctionWithName:@"forward_project_2d_kernel"];
        TORCH_CHECK(function != nil, "Failed to load Metal function");
        
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        TORCH_CHECK(pipeline != nil, "Failed to create compute pipeline: ", error.localizedDescription.UTF8String);
        cached_device = device;
    }
    
    return pipeline;
}

at::Tensor forward_project_2d_mps(
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const at::IntArrayRef output_shape,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
  @autoreleasepool {
    // -------- Validation (unchanged) --------
    TORCH_CHECK(reconstruction.is_mps(),  "Input reconstruction must be on MPS device");
    TORCH_CHECK(rotations.is_mps(),       "Input rotations must be on MPS device");
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic",
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(reconstruction.is_complex(),
                "Reconstruction must be a complex tensor");
    TORCH_CHECK(reconstruction.dim() == 3,
                "Reconstruction must be a 3D tensor (B, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2,
                "Rotations must be (B_rot, P, 2, 2)");

    const auto B            = reconstruction.size(0);
    const auto boxsize      = reconstruction.size(1);
    const auto boxsize_half = reconstruction.size(2);

    const auto B_rot = rotations.size(0);
    const auto P     = rotations.size(1);
    TORCH_CHECK(B_rot == B || B_rot == 1,
                "Batch size of rotations must be 1 or same as reconstruction");

    const auto proj_boxsize      = output_shape[0];
    const auto proj_boxsize_half = output_shape[0] / 2 + 1;

    // -------- Output tensor on MPS (unchanged) --------
    auto projection = torch::zeros({B, P, proj_boxsize, proj_boxsize_half},
                                   torch::TensorOptions()
                                       .dtype(reconstruction.dtype())
                                       .device(reconstruction.device()));

    // -------- Contiguity (unchanged) --------
    auto rec_contiguous  = reconstruction.contiguous();
    auto rot_contiguous  = rotations.contiguous();
    auto proj_contiguous = projection.contiguous();

    // -------- Stream/device from PyTorch --------
    auto* mpsStream = at::mps::getCurrentMPSStream();
    id<MTLDevice> device = mpsStream->device(); // use the stream's device

    // -------- Pipeline for this device (unchanged logic) --------
    id<MTLComputePipelineState> pipeline = get_forward_projection_pipeline(device);

    // -------- Create temporary Metal buffers (unchanged approach) --------
    const size_t rec_bytes  = rec_contiguous.numel()  * rec_contiguous.element_size();
    const size_t rot_bytes  = rot_contiguous.numel()  * rot_contiguous.element_size();
    const size_t proj_bytes = proj_contiguous.numel() * proj_contiguous.element_size();

    id<MTLBuffer> rec_buffer  = [device newBufferWithLength:rec_bytes  options:MTLResourceStorageModeShared];
    id<MTLBuffer> rot_buffer  = [device newBufferWithLength:rot_bytes  options:MTLResourceStorageModeShared];
    id<MTLBuffer> proj_buffer = [device newBufferWithLength:proj_bytes options:MTLResourceStorageModeShared];
    TORCH_CHECK(rec_buffer && rot_buffer && proj_buffer, "Failed to allocate temporary Metal buffers");

    // Copy tensor data to Metal buffers (unchanged)
    auto rec_cpu = rec_contiguous.cpu();
    auto rot_cpu = rot_contiguous.cpu();
    memcpy(rec_buffer.contents, rec_cpu.data_ptr(), rec_bytes);
    memcpy(rot_buffer.contents, rot_cpu.data_ptr(), rot_bytes);

    // -------- Handle shifts tensor (unchanged) --------
    id<MTLBuffer> shifts_buffer = nil;
    bool has_shifts = shifts.has_value();
    int64_t B_shift = 1;
    if (has_shifts) {
      TORCH_CHECK(shifts->is_mps(), "Shifts must be on MPS device");
      TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
      TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1,
                  "Batch size of shifts must be 1 or same as reconstruction");
      TORCH_CHECK(shifts->size(1) == P,
                  "Number of poses in shifts must match rotations");
      B_shift = shifts->size(0);

      auto shifts_contiguous = shifts->contiguous();
      auto shifts_cpu = shifts_contiguous.cpu();
      size_t shifts_bytes = shifts_contiguous.numel() * shifts_contiguous.element_size();
      shifts_buffer = [device newBufferWithLength:shifts_bytes options:MTLResourceStorageModeShared];
      TORCH_CHECK(shifts_buffer, "Failed to allocate shifts Metal buffer");
      memcpy(shifts_buffer.contents, shifts_cpu.data_ptr(), shifts_bytes);
    } else {
      shifts_buffer = [device newBufferWithLength:8 options:MTLResourceStorageModeShared];
      TORCH_CHECK(shifts_buffer, "Failed to allocate placeholder shifts buffer");
      memset(shifts_buffer.contents, 0, 8);
    }

    // -------- Pack parameters (unchanged) --------
    struct Params {
      int B, P, boxsize, boxsize_half;
      int proj_boxsize, proj_boxsize_half, B_rot, B_shift;
      int has_shifts;
      int interpolation_method;
      float oversampling;
      float fourier_radius_cutoff;
    } params = {
      (int)B, (int)P, (int)boxsize, (int)boxsize_half,
      (int)proj_boxsize, (int)proj_boxsize_half, (int)B_rot, (int)B_shift,
      (int)has_shifts, (interpolation == "linear") ? 0 : 1,
      static_cast<float>(oversampling),
      static_cast<float>(fourier_radius_cutoff.value_or(proj_boxsize / 2.0f))
    };

    // -------- Encode on the stream's queue; don't end/commit yourself --------
    dispatch_sync(mpsStream->queue(), ^(){
      id<MTLComputeCommandEncoder> encoder = mpsStream->commandEncoder();
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:rec_buffer    offset:0 atIndex:0];
      [encoder setBuffer:rot_buffer    offset:0 atIndex:1];
      [encoder setBuffer:shifts_buffer offset:0 atIndex:2];
      [encoder setBuffer:proj_buffer   offset:0 atIndex:3];
      [encoder setBytes:&params length:sizeof(Params) atIndex:4];

      // Same launch geometry you had before
      MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
      MTLSize threadgroupsPerGrid   = MTLSizeMake((NSUInteger)P, (NSUInteger)B, 1);
      [encoder dispatchThreadgroups:threadgroupsPerGrid
            threadsPerThreadgroup:threadsPerThreadgroup];

      // IMPORTANT: do NOT call [encoder endEncoding] here; the stream manages it.
    });

    // We still need the results on CPU for your memcpy path; force completion.
    mpsStream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);  // Wait for all queued work on this stream to finish.

    // -------- Copy result back to MPS tensor (unchanged path) --------
    auto proj_cpu = torch::zeros_like(proj_contiguous).cpu();
    memcpy(proj_cpu.data_ptr(), proj_buffer.contents, proj_bytes);
    proj_contiguous.copy_(proj_cpu.to(reconstruction.device()));

    return proj_contiguous;
  } // @autoreleasepool
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_project_2d_mps(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
    TORCH_CHECK(grad_projections.is_mps(), "Input grad_projections must be on MPS device");
    TORCH_CHECK(reconstruction.is_mps(), "Input reconstruction must be on MPS device");
    TORCH_CHECK(rotations.is_mps(), "Input rotations must be on MPS device");
    
    // TODO: Implement MPS backward projection
    // For now, fall back to CPU implementation
    auto cpu_grad_projections = grad_projections.cpu();
    auto cpu_reconstruction = reconstruction.cpu();
    cpu_reconstruction.set_requires_grad(reconstruction.requires_grad());
    auto cpu_rotations = rotations.cpu();
    cpu_rotations.set_requires_grad(rotations.requires_grad());
    c10::optional<at::Tensor> cpu_shifts;
    if (shifts.has_value()) {
        auto cpu_shifts_tensor = shifts.value().cpu();
        // Preserve requires_grad flag which gets lost during device conversion
        cpu_shifts_tensor.set_requires_grad(shifts.value().requires_grad());
        cpu_shifts = cpu_shifts_tensor;
    }
    
    // Forward declare CPU function
    extern std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_project_2d_cpu(
        const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::optional<at::Tensor>&,
        const std::string&, const double, const c10::optional<double>&
    );
    
    auto [grad_reconstruction, grad_rotations, grad_shifts] = backward_project_2d_cpu(
        cpu_grad_projections, cpu_reconstruction, cpu_rotations, cpu_shifts,
        interpolation, oversampling, fourier_radius_cutoff
    );
    
    auto device = reconstruction.device();
    return std::make_tuple(
        grad_reconstruction.to(device),
        grad_rotations.to(device),
        shifts.has_value() ? grad_shifts.to(device) : grad_shifts
    );
}

#endif // __APPLE__