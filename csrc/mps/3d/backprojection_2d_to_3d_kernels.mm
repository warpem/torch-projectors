#include "backprojection_2d_to_3d_kernels.h"

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

// Get Metal compute pipeline for forward 2D->3D backprojection
id<MTLComputePipelineState> get_forward_backproject_3d_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipeline = nil;
    static id<MTLDevice> cached_device = nil;
    static NSString* shaderSource = nil;
    
    if (!shaderSource) {
        shaderSource = [NSString stringWithUTF8String:BACKPROJECT_3D_KERNEL_SOURCE];
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
        
        id<MTLFunction> function = [library newFunctionWithName:@"backproject_2d_to_3d_forw_kernel"];
        TORCH_CHECK(function != nil, "Failed to load Metal function");
        
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        TORCH_CHECK(pipeline != nil, "Failed to create compute pipeline: ", error.localizedDescription.UTF8String);
        cached_device = device;
    }
    
    return pipeline;
}

// -------------------- forward 2D->3D backproject op --------------------
std::tuple<at::Tensor, at::Tensor> backproject_2d_to_3d_forw_mps(
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
  @autoreleasepool {
    // -------- Validation --------
    TORCH_CHECK(projections.is_mps(), "Input projections must be on MPS device");
    TORCH_CHECK(rotations.is_mps(), "Input rotations must be on MPS device");
    TORCH_CHECK(projections.is_complex(), "Projections must be a complex tensor (complex64)");
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic",
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(projections.dim() == 4,
                "Projections must be a 4D tensor (B, P, height, width/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 3 && rotations.size(3) == 3,
                "Rotations must be (B_rot, P, 3, 3)");
    TORCH_CHECK(rotations.scalar_type() == at::kFloat, "Rotations must be float32 on MPS");

    // Validate optional weights
    if (weights.has_value()) {
        TORCH_CHECK(weights->is_mps(), "Weights must be on MPS device");
        TORCH_CHECK(weights->is_floating_point(), "Weights must be a real-valued tensor");
        TORCH_CHECK(weights->dim() == 4, "Weights must be (B, P, height, width/2+1)");
        TORCH_CHECK(weights->sizes() == projections.sizes(), "Weights and projections must have the same shape");
        TORCH_CHECK(weights->scalar_type() == at::kFloat, "Weights must be float32 on MPS");
    }

    const auto B = projections.size(0);
    const auto P = projections.size(1); 
    const auto proj_boxsize = projections.size(2);
    const auto proj_boxsize_half = projections.size(3);

    const auto B_rot = rotations.size(0);
    TORCH_CHECK(B_rot == B || B_rot == 1,
                "Batch size of rotations must be 1 or same as projections");

    // For cubic volumes, depth = height = width, inferred from projection dimensions
    const auto rec_depth = proj_boxsize;
    const auto rec_boxsize = proj_boxsize;
    const auto rec_boxsize_half = proj_boxsize_half;

    // -------- Output tensors on MPS - 3D reconstructions (B, D, H, W/2+1) --------
    auto data_reconstruction = at::zeros({B, rec_depth, rec_boxsize, rec_boxsize_half}, projections.options());
    
    at::Tensor weight_reconstruction;
    bool has_weights = weights.has_value();
    if (has_weights) {
        weight_reconstruction = at::zeros({B, rec_depth, rec_boxsize, rec_boxsize_half}, weights->options());
    } else {
        // Create empty tensor for consistency
        weight_reconstruction = at::empty({0}, projections.options().dtype(rotations.scalar_type()));
    }

    // -------- Ensure contiguity (views handled via storage_offset) --------
    auto proj_contiguous  = projections.is_contiguous()  ? projections  : projections.contiguous();
    auto rot_contiguous   = rotations.is_contiguous()    ? rotations    : rotations.contiguous();
    auto data_rec_contiguous = data_reconstruction.is_contiguous() ? data_reconstruction : data_reconstruction.contiguous();

    // -------- Shifts (optional) --------
    c10::optional<at::Tensor> shifts_contiguous;
    int64_t B_shift = 1;
    if (shifts.has_value()) {
      TORCH_CHECK(shifts->is_mps(), "Shifts must be on MPS device");
      TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
      TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1,
                  "Batch size of shifts must be 1 or same as projections");
      TORCH_CHECK(shifts->size(1) == P,
                  "Number of poses in shifts must match rotations");
      TORCH_CHECK(shifts->scalar_type() == at::kFloat, "Shifts must be float32 on MPS");
      B_shift = shifts->size(0);
      shifts_contiguous = shifts->is_contiguous() ? *shifts : shifts->contiguous();
    }

    // -------- Weights (optional) --------
    c10::optional<at::Tensor> weights_contiguous;
    c10::optional<at::Tensor> weight_rec_contiguous;
    if (has_weights) {
        weights_contiguous = weights->is_contiguous() ? *weights : weights->contiguous();
        weight_rec_contiguous = weight_reconstruction.is_contiguous() ? weight_reconstruction : weight_reconstruction.contiguous();
    }

    // -------- Acquire PyTorch's MPS command buffer & serial queue --------
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");
    dispatch_queue_t serialQueue = (dispatch_queue_t)torch::mps::get_dispatch_queue();
    TORCH_CHECK(serialQueue, "Failed to get MPS dispatch queue");

    // Use the same device as the command buffer (avoids device mismatches)
    id<MTLDevice> device = [commandBuffer device];
    id<MTLComputePipelineState> pipeline = get_forward_backproject_3d_pipeline(device);

    // -------- Kernel params --------
    struct Params3D {
      int B, P, boxsize, boxsize_half;
      int proj_boxsize, proj_boxsize_half, B_rot, B_shift;
      int has_shifts;
      int interpolation_method; // 0=linear, 1=cubic
      float oversampling;
      float fourier_radius_cutoff;
    } params = {
      (int)B, (int)P, (int)rec_boxsize, (int)rec_boxsize_half,
      (int)proj_boxsize, (int)proj_boxsize_half, (int)B_rot, (int)B_shift,
      (int)(shifts.has_value()), (interpolation == "linear") ? 0 : 1,
      static_cast<float>(oversampling),
      static_cast<float>(fourier_radius_cutoff.value_or(proj_boxsize / 2.0f))
    };

    torch::mps::commit();

    // -------- Encode & commit --------
    dispatch_sync(serialQueue, ^(){
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create compute encoder");

      [encoder setComputePipelineState:pipeline];

      // zero-copy binds: underlying MTLBuffer + byte offset
      bindTensor(encoder, proj_contiguous, 0);     // projections
      
      if (has_weights) {
        bindTensor(encoder, *weights_contiguous, 1); // weights
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:1];
      }

      bindTensor(encoder, rot_contiguous, 2);      // rotations

      if (shifts.has_value()) {
        bindTensor(encoder, *shifts_contiguous, 3); // shifts
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:3];
      }

      bindTensor(encoder, data_rec_contiguous, 4); // data_reconstruction
      
      if (has_weights) {
        bindTensor(encoder, *weight_rec_contiguous, 5); // weight_reconstruction
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:5];
      }

      [encoder setBytes:&params length:sizeof(Params3D) atIndex:6];

      // Launch geometry (same as projection for 1:1 behavior)
      MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
      MTLSize threadgroupsPerGrid   = MTLSizeMake((NSUInteger)P, (NSUInteger)B, 1);
      [encoder dispatchThreadgroups:threadgroupsPerGrid
            threadsPerThreadgroup:threadsPerThreadgroup];

      [encoder endEncoding];   
      torch::mps::commit();    
    });

    // -------- Copy back if needed --------
    if (!data_reconstruction.is_contiguous()) {
      data_reconstruction.copy_(data_rec_contiguous);
    }
    if (has_weights && !weight_reconstruction.is_contiguous()) {
      weight_reconstruction.copy_(*weight_rec_contiguous);
    }

    return std::make_tuple(data_reconstruction, weight_reconstruction);
  } // @autoreleasepool
}

// Get Metal compute pipeline for backward 2D->3D backprojection
id<MTLComputePipelineState> get_backward_backproject_3d_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipeline = nil;
    static id<MTLDevice> cached_device = nil;
    static NSString* shaderSource = nil;
    
    if (!shaderSource) {
        shaderSource = [NSString stringWithUTF8String:BACKPROJECT_3D_KERNEL_SOURCE];
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
        
        id<MTLFunction> function = [library newFunctionWithName:@"backproject_2d_to_3d_back_kernel"];
        TORCH_CHECK(function != nil, "Failed to load Metal function");
        
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        TORCH_CHECK(pipeline != nil, "Failed to create compute pipeline: ", error.localizedDescription.UTF8String);
        cached_device = device;
    }
    
    return pipeline;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backproject_2d_to_3d_back_mps(
    const at::Tensor& grad_data_rec,
    const c10::optional<at::Tensor>& grad_weight_rec,
    const at::Tensor& projections,
    const c10::optional<at::Tensor>& weights,
    const at::Tensor& rotations,
    const c10::optional<at::Tensor>& shifts,
    const std::string& interpolation,
    const double oversampling,
    const c10::optional<double>& fourier_radius_cutoff
) {
  @autoreleasepool {
    // -------- Validation --------
    TORCH_CHECK(grad_data_rec.is_mps(), "Input grad_data_rec must be on MPS device");
    TORCH_CHECK(projections.is_mps(), "Input projections must be on MPS device");
    TORCH_CHECK(rotations.is_mps(), "Input rotations must be on MPS device");
    TORCH_CHECK(grad_data_rec.is_complex(), "grad_data_rec must be a complex tensor");
    TORCH_CHECK(projections.is_complex(), "projections must be a complex tensor");
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic",
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(grad_data_rec.dim() == 4, "grad_data_rec must be (B, depth, height, width/2+1)");
    TORCH_CHECK(projections.dim() == 4, "projections must be a 4D tensor (B, P, height, width/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 3 && rotations.size(3) == 3,
                "Rotations must be (B_rot, P, 3, 3)");
    TORCH_CHECK(rotations.scalar_type() == at::kFloat, "Rotations must be float32 on MPS");
    
    if (grad_weight_rec.has_value()) {
        TORCH_CHECK(grad_weight_rec->is_mps(), "grad_weight_rec must be on MPS device");
        TORCH_CHECK(grad_weight_rec->is_floating_point(), "grad_weight_rec must be a real-valued tensor");
        TORCH_CHECK(grad_weight_rec->dim() == 4, "grad_weight_rec must be (B, depth, height, width/2+1)");
        TORCH_CHECK(grad_weight_rec->sizes() == grad_data_rec.sizes(), "grad_weight_rec and grad_data_rec must have same shape");
    }

    const auto B = projections.size(0);
    const auto P = projections.size(1);
    const auto proj_boxsize = projections.size(2);
    const auto proj_boxsize_half = projections.size(3);
    
    const auto rec_depth = grad_data_rec.size(1);
    const auto rec_boxsize = grad_data_rec.size(2);
    const auto rec_boxsize_half = grad_data_rec.size(3);
    
    const auto B_rot = rotations.size(0);
    TORCH_CHECK(B_rot == B || B_rot == 1, "Batch size of rotations must be 1 or same as projections");

    // -------- Initialize gradient tensors based on what's needed --------
    auto grad_projections = at::zeros_like(projections);
    
    at::Tensor grad_weights;
    at::Tensor grad_rotations;
    at::Tensor grad_shifts;
    
    const bool need_rotation_grads = rotations.requires_grad();
    const bool need_shift_grads = shifts.has_value() && shifts->requires_grad();
    const bool has_weights = weights.has_value();
    
    if (has_weights) {
        TORCH_CHECK(weights->is_mps(), "weights must be on MPS device");
        grad_weights = at::zeros_like(*weights);
    } else {
        grad_weights = at::empty({0}, projections.options().dtype(rotations.scalar_type()));
    }
    
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
                    "Batch size of shifts must be 1 or same as projections");
        TORCH_CHECK(shifts->size(1) == P, "Number of poses in shifts must match rotations");
        TORCH_CHECK(shifts->scalar_type() == at::kFloat, "Shifts must be float32 on MPS");
        B_shift = shifts->size(0);
        
        if (need_shift_grads) {
            grad_shifts = at::zeros_like(*shifts);
        } else {
            grad_shifts = at::empty({0}, projections.options().dtype(rotations.scalar_type()));
        }
    } else {
        grad_shifts = at::empty({0}, projections.options().dtype(rotations.scalar_type()));
    }

    // -------- Ensure contiguity --------
    auto grad_data_rec_contiguous = grad_data_rec.is_contiguous() ? grad_data_rec : grad_data_rec.contiguous();
    auto proj_contiguous = projections.is_contiguous() ? projections : projections.contiguous();
    auto rot_contiguous = rotations.is_contiguous() ? rotations : rotations.contiguous();
    auto grad_proj_contiguous = grad_projections.is_contiguous() ? grad_projections : grad_projections.contiguous();

    c10::optional<at::Tensor> grad_weight_rec_contiguous;
    c10::optional<at::Tensor> weights_contiguous;
    c10::optional<at::Tensor> grad_weights_contiguous;
    c10::optional<at::Tensor> shifts_contiguous;
    c10::optional<at::Tensor> grad_shifts_contiguous;
    c10::optional<at::Tensor> grad_rot_contiguous;

    if (grad_weight_rec.has_value()) {
        grad_weight_rec_contiguous = grad_weight_rec->is_contiguous() ? *grad_weight_rec : grad_weight_rec->contiguous();
    }
    if (has_weights) {
        weights_contiguous = weights->is_contiguous() ? *weights : weights->contiguous();
        grad_weights_contiguous = grad_weights.is_contiguous() ? grad_weights : grad_weights.contiguous();
    }
    if (shifts.has_value()) {
        shifts_contiguous = shifts->is_contiguous() ? *shifts : shifts->contiguous();
        if (need_shift_grads) {
            grad_shifts_contiguous = grad_shifts.is_contiguous() ? grad_shifts : grad_shifts.contiguous();
        }
    }
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
    id<MTLComputePipelineState> pipeline = get_backward_backproject_3d_pipeline(device);

    // -------- Kernel params with gradient flags --------
    struct Params3D {
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
      bindTensor(encoder, grad_data_rec_contiguous, 0);  // grad_data_rec
      
      if (grad_weight_rec.has_value()) {
        bindTensor(encoder, *grad_weight_rec_contiguous, 1);  // grad_weight_rec
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:1];
      }
      
      bindTensor(encoder, proj_contiguous, 2);           // projections
      
      if (has_weights) {
        bindTensor(encoder, *weights_contiguous, 3);     // weights
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:3];
      }
      
      bindTensor(encoder, rot_contiguous, 4);            // rotations
      
      if (shifts.has_value()) {
        bindTensor(encoder, *shifts_contiguous, 5);      // shifts
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:5];
      }
      
      bindTensor(encoder, grad_proj_contiguous, 6);      // grad_projections
      
      if (has_weights) {
        bindTensor(encoder, *grad_weights_contiguous, 7); // grad_weights
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:7];
      }
      
      if (need_rotation_grads) {
        bindTensor(encoder, *grad_rot_contiguous, 8);    // grad_rotations
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:8];
      }
      
      if (need_shift_grads) {
        bindTensor(encoder, *grad_shifts_contiguous, 9); // grad_shifts
      } else {
        uint64_t zero = 0;
        [encoder setBytes:&zero length:sizeof(zero) atIndex:9];
      }
      
      [encoder setBytes:&params length:sizeof(Params3D) atIndex:10];

      // Launch geometry - same as forward pass
      MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
      MTLSize threadgroupsPerGrid   = MTLSizeMake((NSUInteger)P, (NSUInteger)B, 1);
      [encoder dispatchThreadgroups:threadgroupsPerGrid
            threadsPerThreadgroup:threadsPerThreadgroup];

      [encoder endEncoding];
      torch::mps::commit();
    });

    // -------- Copy back if needed --------
    if (!grad_projections.is_contiguous()) {
      grad_projections.copy_(grad_proj_contiguous);
    }
    if (has_weights && !grad_weights.is_contiguous()) {
      grad_weights.copy_(*grad_weights_contiguous);
    }
    if (need_rotation_grads && !grad_rotations.is_contiguous()) {
      grad_rotations.copy_(*grad_rot_contiguous);
    }
    if (need_shift_grads && !grad_shifts.is_contiguous()) {
      grad_shifts.copy_(*grad_shifts_contiguous);
    }

    return std::make_tuple(grad_projections, grad_weights, grad_rotations, grad_shifts);
  } // @autoreleasepool
}

#endif // __APPLE__