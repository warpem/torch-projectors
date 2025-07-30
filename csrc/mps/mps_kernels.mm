#include "mps_kernels.h"

#ifdef __APPLE__

#include <torch/extension.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <ATen/mps/MPSStream.h>
#include "projection_kernels.h"

// Retrieve the MTLBuffer that backs a tensor's storage.
// NOTE: storage().data() is const void*, so keep const and cast via __bridge.
static inline id<MTLBuffer> getTensorMTLBuffer(const at::Tensor& t) {
  const void* raw = t.storage().data();                 // const void*
  return (__bridge id<MTLBuffer>)(const_cast<void*>(raw));
}

// Bind tensor storage (buffer + byte offset) without copies.
static inline void bindTensor(
    id<MTLComputeCommandEncoder> encoder,
    const at::Tensor& tensor,
    NSUInteger index)
{
  TORCH_CHECK(tensor.is_mps(), "Tensor at index ", (unsigned)index, " must be on MPS device");
  // Ensure a dense layout; views handled via storage_offset()
  const at::Tensor t = tensor.is_contiguous() ? tensor : tensor.contiguous();

  const NSUInteger byte_offset =
      static_cast<NSUInteger>(t.storage_offset()) * t.element_size();

  id<MTLBuffer> buf = getTensorMTLBuffer(t);
  TORCH_CHECK(buf != nil, "Failed to obtain MTLBuffer for tensor (arg ", (unsigned)index, ")");
  [encoder setBuffer:buf offset:byte_offset atIndex:index];
}

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
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.fastMathEnabled = YES;
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
                                                    options:opts
                                                        error:&error];
        TORCH_CHECK(library != nil, "Failed to compile Metal library: ", error.localizedDescription.UTF8String);
        
        id<MTLFunction> function = [library newFunctionWithName:@"forward_project_2d_kernel"];
        TORCH_CHECK(function != nil, "Failed to load Metal function");
        
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        TORCH_CHECK(pipeline != nil, "Failed to create compute pipeline: ", error.localizedDescription.UTF8String);
        cached_device = device;
    }
    
    return pipeline;
}

// -------------------- forward op (zero-copy; stream-managed) --------------------
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
    // -------- Validation --------
    TORCH_CHECK(reconstruction.is_mps(),  "Input reconstruction must be on MPS device");
    TORCH_CHECK(rotations.is_mps(),       "Input rotations must be on MPS device");
    TORCH_CHECK(reconstruction.is_complex(), "Reconstruction must be a complex tensor (complex64)");
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic",
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(reconstruction.dim() == 3,
                "Reconstruction must be a 3D tensor (B, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2,
                "Rotations must be (B_rot, P, 2, 2)");
    TORCH_CHECK(rotations.scalar_type() == at::kFloat, "Rotations must be float32 on MPS");

    const auto B            = reconstruction.size(0);
    const auto boxsize      = reconstruction.size(1);
    const auto boxsize_half = reconstruction.size(2);

    const auto B_rot = rotations.size(0);
    const auto P     = rotations.size(1);
    TORCH_CHECK(B_rot == B || B_rot == 1,
                "Batch size of rotations must be 1 or same as reconstruction");

    const auto proj_boxsize      = output_shape[0];
    const auto proj_boxsize_half = output_shape[0] / 2 + 1;

    // -------- Output tensor on MPS --------
    auto projection = at::zeros({B, P, proj_boxsize, proj_boxsize_half},
                                reconstruction.options());

    // -------- Ensure contiguity (views handled via storage_offset) --------
    auto rec_contiguous  = reconstruction.is_contiguous()  ? reconstruction  : reconstruction.contiguous();
    auto rot_contiguous  = rotations.is_contiguous()       ? rotations       : rotations.contiguous();
    auto proj_contiguous = projection.is_contiguous()      ? projection      : projection.contiguous();

    // -------- Shifts (optional) --------
    c10::optional<at::Tensor> shifts_contiguous;
    int64_t B_shift = 1;
    const bool has_shifts = shifts.has_value();
    if (has_shifts) {
      TORCH_CHECK(shifts->is_mps(), "Shifts must be on MPS device");
      TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
      TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1,
                  "Batch size of shifts must be 1 or same as reconstruction");
      TORCH_CHECK(shifts->size(1) == P,
                  "Number of poses in shifts must match rotations");
      TORCH_CHECK(shifts->scalar_type() == at::kFloat, "Shifts must be float32 on MPS");
      B_shift = shifts->size(0);
      shifts_contiguous = shifts->is_contiguous() ? *shifts : shifts->contiguous();
    }

    // -------- Acquire PyTorch's MPS command buffer & serial queue --------
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");
    dispatch_queue_t serialQueue = (dispatch_queue_t)torch::mps::get_dispatch_queue();
    TORCH_CHECK(serialQueue, "Failed to get MPS dispatch queue");

    // Use the same device as the command buffer (avoids device mismatches)
    id<MTLDevice> device = [commandBuffer device];
    id<MTLComputePipelineState> pipeline = get_forward_projection_pipeline(device);

    // -------- Kernel params --------
    struct Params {
      int B, P, boxsize, boxsize_half;
      int proj_boxsize, proj_boxsize_half, B_rot, B_shift;
      int has_shifts;
      int interpolation_method; // 0=linear, 1=cubic
      float oversampling;
      float fourier_radius_cutoff;
    } params = {
      (int)B, (int)P, (int)boxsize, (int)boxsize_half,
      (int)proj_boxsize, (int)proj_boxsize_half, (int)B_rot, (int)B_shift,
      (int)has_shifts, (interpolation == "linear") ? 0 : 1,
      static_cast<float>(oversampling),
      static_cast<float>(fourier_radius_cutoff.value_or(proj_boxsize / 2.0f))
    };

    // -------- Encode & commit (you own encoder lifecycle in this API) --------
    dispatch_sync(serialQueue, ^(){
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create compute encoder");

      [encoder setComputePipelineState:pipeline];

      // zero-copy binds: underlying MTLBuffer + byte offset
      bindTensor(encoder, rec_contiguous, 0);
      bindTensor(encoder, rot_contiguous, 1);

      if (has_shifts) {
        bindTensor(encoder, *shifts_contiguous, 2);
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:2];
      }

      bindTensor(encoder, proj_contiguous, 3);
      [encoder setBytes:&params length:sizeof(Params) atIndex:4];

      // Launch geometry (same as before for 1:1 behavior)
      MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
      MTLSize threadgroupsPerGrid   = MTLSizeMake((NSUInteger)P, (NSUInteger)B, 1);
      [encoder dispatchThreadgroups:threadgroupsPerGrid
            threadsPerThreadgroup:threadsPerThreadgroup];

      [encoder endEncoding];   // We manage encoder lifecycle here
      torch::mps::commit();    // Submit the command buffer (async)
      // If you need immediate CPU readback after this call, add:
      // torch::mps::synchronize();
    });

    // If 'projection' wasn't contiguous, mirror back (still on MPS)
    if (!projection.is_contiguous()) {
      projection.copy_(proj_contiguous);
      return projection;
    }
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