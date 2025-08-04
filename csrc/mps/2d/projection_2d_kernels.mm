#include "projection_2d_kernels.h"

#ifdef __APPLE__

#include <torch/extension.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <ATen/mps/MPSStream.h>
#include "../projection_kernels.h"

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
at::Tensor project_2d_forw_mps(
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

    torch::mps::commit();

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


// Get Metal compute pipeline for backward projection
id<MTLComputePipelineState> get_backward_projection_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipeline = nil;
    static id<MTLDevice> cached_device = nil;
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
        
        id<MTLFunction> function = [library newFunctionWithName:@"backward_project_2d_kernel"];
        TORCH_CHECK(function != nil, "Failed to load Metal function");
        
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        TORCH_CHECK(pipeline != nil, "Failed to create compute pipeline: ", error.localizedDescription.UTF8String);
        cached_device = device;
    }
    
    return pipeline;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> project_2d_back_mps(
    const at::Tensor& grad_projections,
    const at::Tensor& reconstruction,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
  @autoreleasepool {
    // -------- Validation --------
    TORCH_CHECK(grad_projections.is_mps(), "Input grad_projections must be on MPS device");
    TORCH_CHECK(reconstruction.is_mps(), "Input reconstruction must be on MPS device");
    TORCH_CHECK(rotations.is_mps(), "Input rotations must be on MPS device");
    TORCH_CHECK(grad_projections.is_complex(), "Grad projections must be a complex tensor");
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic",
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(grad_projections.dim() == 4, "Grad projections must be a 4D tensor (B, P, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2,
                "Rotations must be (B_rot, P, 2, 2)");
    TORCH_CHECK(rotations.scalar_type() == at::kFloat, "Rotations must be float32 on MPS");

    const auto B = grad_projections.size(0);
    const auto P = grad_projections.size(1);
    const auto proj_boxsize = grad_projections.size(2);
    const auto proj_boxsize_half = grad_projections.size(3);
    
    const auto rec_boxsize = reconstruction.size(1);
    const auto rec_boxsize_half = reconstruction.size(2);
    
    const auto B_rot = rotations.size(0);
    TORCH_CHECK(B_rot == B || B_rot == 1, "Batch size of rotations must be 1 or same as reconstruction");

    // -------- Initialize gradient tensors based on what's needed --------
    auto grad_reconstruction = at::zeros({B, rec_boxsize, rec_boxsize_half}, grad_projections.options());
    
    at::Tensor grad_rotations;
    at::Tensor grad_shifts;
    
    const bool need_rotation_grads = rotations.requires_grad();
    const bool need_shift_grads = shifts.has_value() && shifts->requires_grad();
    
    if (need_rotation_grads) {
        grad_rotations = at::zeros_like(rotations);
    } else {
        grad_rotations = at::empty({0}, rotations.options());
    }

    int64_t B_shift = 1;
    if (shifts.has_value()) {
        TORCH_CHECK(shifts->is_mps(), "Shifts must be on MPS device");
        TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
        TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1,
                    "Batch size of shifts must be 1 or same as reconstruction");
        TORCH_CHECK(shifts->size(1) == P, "Number of poses in shifts must match rotations");
        TORCH_CHECK(shifts->scalar_type() == at::kFloat, "Shifts must be float32 on MPS");
        B_shift = shifts->size(0);
        
        if (need_shift_grads) {
            grad_shifts = at::zeros_like(*shifts);
        } else {
            grad_shifts = at::empty({0}, grad_projections.options().dtype(rotations.scalar_type()));
        }
    } else {
        grad_shifts = at::empty({0}, grad_projections.options().dtype(rotations.scalar_type()));
    }

    // -------- Ensure contiguity --------
    auto grad_proj_contiguous = grad_projections.is_contiguous() ? grad_projections : grad_projections.contiguous();
    auto rec_contiguous = reconstruction.is_contiguous() ? reconstruction : reconstruction.contiguous();
    auto rot_contiguous = rotations.is_contiguous() ? rotations : rotations.contiguous();
    auto grad_rec_contiguous = grad_reconstruction.is_contiguous() ? grad_reconstruction : grad_reconstruction.contiguous();

    c10::optional<at::Tensor> shifts_contiguous;
    c10::optional<at::Tensor> grad_shifts_contiguous;
    if (shifts.has_value()) {
        shifts_contiguous = shifts->is_contiguous() ? *shifts : shifts->contiguous();
        if (need_shift_grads) {
            grad_shifts_contiguous = grad_shifts.is_contiguous() ? grad_shifts : grad_shifts.contiguous();
        }
    }

    c10::optional<at::Tensor> grad_rot_contiguous;
    if (need_rotation_grads) {
        grad_rot_contiguous = grad_rotations.is_contiguous() ? grad_rotations : grad_rotations.contiguous();
    }

    torch::mps::commit();

    // -------- Acquire PyTorch's MPS command buffer & serial queue --------
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");
    dispatch_queue_t serialQueue = (dispatch_queue_t)torch::mps::get_dispatch_queue();
    TORCH_CHECK(serialQueue, "Failed to get MPS dispatch queue");

    id<MTLDevice> device = [commandBuffer device];
    id<MTLComputePipelineState> pipeline = get_backward_projection_pipeline(device);

    // -------- Kernel params with gradient flags --------
    struct Params {
      int B, P, boxsize, boxsize_half;
      int proj_boxsize, proj_boxsize_half, B_rot, B_shift;
      int has_shifts;
      int interpolation_method; // Encode gradient flags in upper bits
      float oversampling;
      float fourier_radius_cutoff;
    } params = {
      (int)B, (int)P, (int)rec_boxsize, (int)rec_boxsize_half,
      (int)proj_boxsize, (int)proj_boxsize_half, (int)B_rot, (int)B_shift,
      (int)(shifts.has_value()),
      (interpolation == "linear" ? 0 : 1) | 
      (need_rotation_grads ? 0x10 : 0) | 
      (need_shift_grads ? 0x20 : 0),  // Pack flags into upper bits
      static_cast<float>(oversampling),
      static_cast<float>(fourier_radius_cutoff.value_or(proj_boxsize / 2.0))
    };

    // -------- Execute kernel --------
    dispatch_sync(serialQueue, ^(){
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create compute encoder");

      [encoder setComputePipelineState:pipeline];

      // Bind tensors
      bindTensor(encoder, grad_proj_contiguous, 0);  // grad_projections
      bindTensor(encoder, rec_contiguous, 1);        // reconstruction
      bindTensor(encoder, rot_contiguous, 2);        // rotations
      
      if (shifts.has_value()) {
        bindTensor(encoder, *shifts_contiguous, 3);  // shifts
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:3];
      }
      
      bindTensor(encoder, grad_rec_contiguous, 4);   // grad_reconstruction
      
      if (need_rotation_grads) {
        bindTensor(encoder, *grad_rot_contiguous, 5);  // grad_rotations
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:5];
      }
      
      if (need_shift_grads) {
        bindTensor(encoder, *grad_shifts_contiguous, 6);  // grad_shifts
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:6];
      }
      
      [encoder setBytes:&params length:sizeof(Params) atIndex:7];

      // Launch geometry - same as forward pass
      MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
      MTLSize threadgroupsPerGrid   = MTLSizeMake((NSUInteger)P, (NSUInteger)B, 1);
      [encoder dispatchThreadgroups:threadgroupsPerGrid
            threadsPerThreadgroup:threadsPerThreadgroup];

      [encoder endEncoding];
      torch::mps::commit();
    });

    // -------- Copy back if needed --------
    if (!grad_reconstruction.is_contiguous()) {
      grad_reconstruction.copy_(grad_rec_contiguous);
    }
    if (need_rotation_grads && !grad_rotations.is_contiguous()) {
      grad_rotations.copy_(*grad_rot_contiguous);
    }
    if (need_shift_grads && !grad_shifts.is_contiguous()) {
      grad_shifts.copy_(*grad_shifts_contiguous);
    }

    return std::make_tuple(grad_reconstruction, grad_rotations, grad_shifts);
  } // @autoreleasepool
}

#endif // __APPLE__