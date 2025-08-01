# torch_projectors/ops.py

import torch
from typing import Optional

def forward_project_2d(
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts: Optional[torch.Tensor] = None,
    output_shape: Optional[tuple[int, ...]] = None,
    interpolation: str = 'linear',
    oversampling: float = 1.0,
    fourier_radius_cutoff: Optional[float] = None
) -> torch.Tensor:
    # Validate reconstruction dimensions
    if reconstruction.dim() < 2:
        raise ValueError("Reconstruction must have at least 2 dimensions")
    
    boxsize = reconstruction.shape[-2] 
    boxsize_half = reconstruction.shape[-1]
    
    # Enforce square, even dimensions
    if boxsize % 2 != 0:
        raise ValueError(f"Boxsize ({boxsize}) must be even. Only even dimensions are supported.")
    if boxsize != (boxsize_half - 1) * 2:
        raise ValueError(f"Reconstruction shape mismatch: expected boxsize {boxsize} to match 2*(boxsize_half-1) = {(boxsize_half - 1) * 2}")
    
    # Set default output shape to square
    if output_shape is None:
        output_shape = (boxsize, boxsize)
    else:
        # Validate output shape is also square and even
        if len(output_shape) != 2 or output_shape[0] != output_shape[1]:
            raise ValueError(f"Output shape {output_shape} must be square (height == width)")
        if output_shape[0] % 2 != 0:
            raise ValueError(f"Output boxsize {output_shape[0]} must be even")
    
    return _ForwardProject2D.apply(reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff)

def backward_project_2d(
    projections: torch.Tensor,
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts: Optional[torch.Tensor] = None,
    interpolation: str = 'linear',
    oversampling: float = 1.0
) -> torch.Tensor:
    """
    Backward project 2D projections to reconstruction (returns only reconstruction gradients)
    
    This function computes only the reconstruction gradients from the unified backward
    projection operation, discarding rotation and shift gradients for convenience.
    """
    # Call the unified backward function and return only reconstruction gradients
    grad_reconstruction, _, _ = torch.ops.torch_projectors.backward_project_2d(
        projections, reconstruction, rotations, shifts, interpolation, oversampling, None
    )
    
    return grad_reconstruction

def forward_project_3d_to_2d(
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts: Optional[torch.Tensor] = None,
    output_shape: Optional[tuple[int, ...]] = None,
    interpolation: str = 'linear',
    oversampling: float = 1.0,
    fourier_radius_cutoff: Optional[float] = None
) -> torch.Tensor:
    """
    Forward project 3D Fourier volume to 2D projections using the central slice theorem.
    
    This function projects 4D Fourier-space reconstructions [B, D, H, W/2+1] to 2D projections
    [B, P, H_out, W_out/2+1] by sampling the 3D volume at rotated coordinates corresponding
    to central slices through the origin.
    
    Args:
        reconstruction: 4D complex tensor [B, D, H, W/2+1] - 3D Fourier volume in FFTW format
        rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices
        shifts: Optional 3D real tensor [B_shift, P, 2] - 2D translation shifts
        output_shape: Tuple (H_out, W_out) for output projection size. Defaults to (H, W)
        interpolation: 'linear' (trilinear) or 'cubic' (tricubic) interpolation
        oversampling: Coordinate scaling factor (>1 for oversampling)
        fourier_radius_cutoff: Optional frequency cutoff for low-pass filtering
        
    Returns:
        4D complex tensor [B, P, H_out, W_out/2+1] - the 2D projections
        
    Note:
        - Uses central slice theorem: 2D projection = central slice through 3D Fourier volume
        - Supports batch broadcasting: B_rot and B_shift can be 1 or match B
        - 3D Friedel symmetry is automatically handled for real-valued reconstructions
    """
    # Validate reconstruction dimensions
    if reconstruction.dim() != 4:
        raise ValueError("Reconstruction must be a 4D tensor (B, D, H, W/2+1)")
    
    B, D, boxsize, boxsize_half = reconstruction.shape
    
    # Enforce square, even dimensions
    if boxsize % 2 != 0:
        raise ValueError(f"Boxsize ({boxsize}) must be even. Only even dimensions are supported.")
    if boxsize != (boxsize_half - 1) * 2:
        raise ValueError(f"Reconstruction shape mismatch: expected boxsize {boxsize} to match 2*(boxsize_half-1) = {(boxsize_half - 1) * 2}")
    
    # Set default output shape to square
    if output_shape is None:
        output_shape = (boxsize, boxsize)
    else:
        # Validate output shape is also square and even
        if len(output_shape) != 2 or output_shape[0] != output_shape[1]:
            raise ValueError(f"Output shape {output_shape} must be square (height == width)")
        if output_shape[0] % 2 != 0:
            raise ValueError(f"Output boxsize {output_shape[0]} must be even")
    
    return _ForwardProject3DTo2D.apply(reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff)

def backward_project_3d_to_2d(
    projections: torch.Tensor,
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts: Optional[torch.Tensor] = None,
    interpolation: str = 'linear',
    oversampling: float = 1.0
) -> torch.Tensor:
    """
    Backward project 2D projections to 3D reconstruction (returns only reconstruction gradients)
    
    This function computes only the 3D reconstruction gradients from the unified backward
    projection operation, discarding rotation and shift gradients for convenience.
    
    Args:
        projections: 4D complex tensor [B, P, H, W/2+1] - 2D projections  
        reconstruction: 4D complex tensor [B, D, H, W/2+1] - 3D Fourier volume
        rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices
        shifts: Optional 3D real tensor [B_shift, P, 2] - 2D translation shifts
        interpolation: 'linear' (trilinear) or 'cubic' (tricubic) interpolation
        oversampling: Coordinate scaling factor (must match forward pass)
        
    Returns:
        4D complex tensor [B, D, H, W/2+1] - gradients w.r.t. 3D reconstruction
    """
    # Call the unified backward function and return only reconstruction gradients
    grad_reconstruction, _, _ = torch.ops.torch_projectors.backward_project_3d_to_2d(
        projections, reconstruction, rotations, shifts, interpolation, oversampling, None
    )
    
    return grad_reconstruction

# --- Autograd Registration ---

class _ForwardProject2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff):
        # Unsqueeze if single reconstruction/pose set
        if reconstruction.dim() == 2:
            reconstruction = reconstruction.unsqueeze(0)
        if rotations.dim() == 2:
            rotations = rotations.unsqueeze(0)
        if rotations.dim() == 3:
            rotations = rotations.unsqueeze(0)
        if shifts is not None:
            if shifts.dim() == 1:
                shifts = shifts.unsqueeze(0)
            if shifts.dim() == 2:
                shifts = shifts.unsqueeze(0)

        # Let C++ handle batch size mismatches (1 vs N)
        if rotations.size(0) != reconstruction.size(0) and rotations.size(0) != 1:
            raise ValueError("Batch size of rotations must be 1 or match reconstruction")
        
        projection = torch.ops.torch_projectors.forward_project_2d(reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff)
        
        ctx.save_for_backward(reconstruction, rotations, shifts)
        ctx.interpolation = interpolation
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        return projection

    @staticmethod
    def backward(ctx, grad_output):
        reconstruction, rotations, shifts = ctx.saved_tensors
        
        grad_reconstruction, grad_rotations, grad_shifts = torch.ops.torch_projectors.backward_project_2d(
            grad_output.contiguous(),
            reconstruction,
            rotations,
            shifts,
            ctx.interpolation,
            ctx.oversampling,
            ctx.fourier_radius_cutoff
        )
        
        # If shifts was None in the forward pass, return None for its gradient
        if shifts is None:
            grad_shifts = None
        
        return grad_reconstruction, grad_rotations, grad_shifts, None, None, None, None

class _ForwardProject3DTo2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff):
        # Unsqueeze if single reconstruction/pose set
        if reconstruction.dim() == 3:
            reconstruction = reconstruction.unsqueeze(0)
        if rotations.dim() == 2:
            rotations = rotations.unsqueeze(0)
        if rotations.dim() == 3:
            rotations = rotations.unsqueeze(0)
        if shifts is not None:
            if shifts.dim() == 1:
                shifts = shifts.unsqueeze(0)
            if shifts.dim() == 2:
                shifts = shifts.unsqueeze(0)

        # Let C++ handle batch size mismatches (1 vs N)
        if rotations.size(0) != reconstruction.size(0) and rotations.size(0) != 1:
            raise ValueError("Batch size of rotations must be 1 or match reconstruction")
        
        projection = torch.ops.torch_projectors.forward_project_3d_to_2d(reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff)
        
        ctx.save_for_backward(reconstruction, rotations, shifts)
        ctx.interpolation = interpolation
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        return projection

    @staticmethod
    def backward(ctx, grad_output):
        reconstruction, rotations, shifts = ctx.saved_tensors
        
        grad_reconstruction, grad_rotations, grad_shifts = torch.ops.torch_projectors.backward_project_3d_to_2d(
            grad_output.contiguous(),
            reconstruction,
            rotations,
            shifts,
            ctx.interpolation,
            ctx.oversampling,
            ctx.fourier_radius_cutoff
        )
        
        # If shifts was None in the forward pass, return None for its gradient
        if shifts is None:
            grad_shifts = None
        
        return grad_reconstruction, grad_rotations, grad_shifts, None, None, None, None

def _backward_project_2d_backward(ctx, grad_output):
    # No-op placeholder
    return None

def _backward_project_3d_to_2d_backward(ctx, grad_output):
    # No-op placeholder
    return None

torch.library.register_autograd(
    "torch_projectors::backward_project_2d", 
    _backward_project_2d_backward,
    setup_context=lambda ctx, inputs, output: None
)

torch.library.register_autograd(
    "torch_projectors::backward_project_3d_to_2d", 
    _backward_project_3d_to_2d_backward,
    setup_context=lambda ctx, inputs, output: None
) 