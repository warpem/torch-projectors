#include "mps_kernels.h"

#ifdef __APPLE__

#include <torch/extension.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
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
    TORCH_CHECK(reconstruction.is_mps(), "Input reconstruction must be on MPS device");
    TORCH_CHECK(rotations.is_mps(), "Input rotations must be on MPS device");
    TORCH_CHECK(interpolation == "linear" || interpolation == "cubic", 
                "Supported interpolation methods: 'linear', 'cubic'");
    TORCH_CHECK(reconstruction.is_complex(), "Reconstruction must be a complex tensor");
    TORCH_CHECK(reconstruction.dim() == 3, "Reconstruction must be a 3D tensor (B, boxsize, boxsize/2+1)");
    TORCH_CHECK(rotations.dim() == 4 && rotations.size(2) == 2 && rotations.size(3) == 2, "Rotations must be (B_rot, P, 2, 2)");
    
    // Extract tensor dimensions
    const auto B = reconstruction.size(0);
    const auto boxsize = reconstruction.size(1);
    const auto boxsize_half = reconstruction.size(2);
    
    const auto B_rot = rotations.size(0);
    const auto P = rotations.size(1);
    TORCH_CHECK(B_rot == B || B_rot == 1, "Batch size of rotations must be 1 or same as reconstruction");
    
    const auto proj_boxsize = output_shape[0];
    const auto proj_boxsize_half = output_shape[0] / 2 + 1;
    
    // Create output tensor on MPS device using PyTorch (not C++)
    auto projection = torch::zeros({B, P, proj_boxsize, proj_boxsize_half}, 
                                   torch::TensorOptions()
                                       .dtype(reconstruction.dtype())
                                       .device(reconstruction.device()));
    
    // Ensure tensors are contiguous
    auto rec_contiguous = reconstruction.contiguous();
    auto rot_contiguous = rotations.contiguous();
    auto proj_contiguous = projection.contiguous();
    
    // Get Metal device and create command queue/buffer  
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Get the compute pipeline
    id<MTLComputePipelineState> pipeline = get_forward_projection_pipeline(device);
    [encoder setComputePipelineState:pipeline];
    
    // For MPS tensors, we need to copy data to Metal buffers we can control
    // This is safer than trying to wrap the MPS storage directly
    size_t rec_bytes = rec_contiguous.numel() * rec_contiguous.element_size();
    size_t rot_bytes = rot_contiguous.numel() * rot_contiguous.element_size();
    size_t proj_bytes = proj_contiguous.numel() * proj_contiguous.element_size();
    
    id<MTLBuffer> rec_buffer = [device newBufferWithLength:rec_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> rot_buffer = [device newBufferWithLength:rot_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> proj_buffer = [device newBufferWithLength:proj_bytes options:MTLResourceStorageModeShared];
    
    // Copy tensor data to Metal buffers
    auto rec_cpu = rec_contiguous.cpu();
    auto rot_cpu = rot_contiguous.cpu();
    memcpy(rec_buffer.contents, rec_cpu.data_ptr(), rec_bytes);
    memcpy(rot_buffer.contents, rot_cpu.data_ptr(), rot_bytes);
    
    [encoder setBuffer:rec_buffer offset:0 atIndex:0];
    [encoder setBuffer:rot_buffer offset:0 atIndex:1];
    
    // Handle shifts tensor
    id<MTLBuffer> shifts_buffer = nil;
    bool has_shifts = shifts.has_value();
    int64_t B_shift = 1;
    if (has_shifts) {
        TORCH_CHECK(shifts->is_mps(), "Shifts must be on MPS device");
        TORCH_CHECK(shifts->dim() == 3, "Shifts must be (B_shift, P, 2)");
        TORCH_CHECK(shifts->size(0) == B || shifts->size(0) == 1, "Batch size of shifts must be 1 or same as reconstruction");
        TORCH_CHECK(shifts->size(1) == P, "Number of poses in shifts must match rotations");
        B_shift = shifts->size(0);
        
        auto shifts_contiguous = shifts->contiguous();
        auto shifts_cpu = shifts_contiguous.cpu();
        size_t shifts_bytes = shifts_contiguous.numel() * shifts_contiguous.element_size();
        shifts_buffer = [device newBufferWithLength:shifts_bytes options:MTLResourceStorageModeShared];
        memcpy(shifts_buffer.contents, shifts_cpu.data_ptr(), shifts_bytes);
    } else {
        shifts_buffer = [device newBufferWithLength:8 options:MTLResourceStorageModeShared];
    }
    [encoder setBuffer:shifts_buffer offset:0 atIndex:2];
    [encoder setBuffer:proj_buffer offset:0 atIndex:3];
    
    // Pack parameters into struct
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
    [encoder setBytes:&params length:sizeof(Params) atIndex:4];
    
    // Optimize thread organization: each threadgroup handles one entire projection
    // Grid: (P, B, 1) - one threadgroup per projection
    // Threadgroup: (256, 1, 1) - threads loop over pixels within projection
    MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake(
        (NSUInteger)P,  // One threadgroup per pose
        (NSUInteger)B,  // One threadgroup per batch  
        1
    );
    
    // Dispatch compute shader
    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    
    // Execute and wait
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back to MPS tensor
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