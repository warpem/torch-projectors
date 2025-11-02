# torch_projectors/ops.py

import torch
from typing import Optional

def project_2d_forw(
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
    
    return _Project2D.apply(reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff)

def project_2d_back(
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
    grad_reconstruction, _, _ = torch.ops.torch_projectors.project_2d_back(
        projections, reconstruction, rotations, shifts, interpolation, oversampling, None
    )
    
    return grad_reconstruction

def backproject_2d_forw(
    projections: torch.Tensor,
    rotations: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    shifts: Optional[torch.Tensor] = None,
    interpolation: str = 'linear',
    oversampling: float = 1.0,
    fourier_radius_cutoff: Optional[float] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Back-project 2D projections into 2D reconstructions (adjoint/transpose operation)
    
    This function accumulates 2D projection data (and optional weights) into 2D reconstructions.
    It is the mathematical adjoint/transpose of forward projection and supports optional weight
    accumulation for CTF handling or other applications.
    
    Args:
        projections: 4D complex tensor [B, P, height, width/2+1] - 2D projections in FFTW format
        rotations: 4D real tensor [B_rot, P, 2, 2] - 2x2 rotation matrices
        weights: Optional 4D real tensor [B, P, height, width/2+1] - weights (e.g., CTF^2)
        shifts: Optional 3D real tensor [B_shift, P, 2] - 2D translation shifts
        interpolation: 'linear' (bilinear) or 'cubic' (bicubic) interpolation
        oversampling: Coordinate scaling factor (>1 for oversampling)
        fourier_radius_cutoff: Optional frequency cutoff for low-pass filtering
        
    Returns:
        Tuple of (data_reconstruction, weight_reconstruction):
        - data_reconstruction: 3D complex tensor [B, height, width/2+1] - accumulated projection data
        - weight_reconstruction: 3D real tensor [B, height, width/2+1] - accumulated weights
          (empty tensor if weights=None)
        
    Note:
        - This is the adjoint operation of forward_project_2d()
        - Uses conjugate phase shifts for proper mathematical adjoint
        - Supports batch broadcasting: B_rot and B_shift can be 1 or match B
        - Weight accumulation enables Wiener-like filtering in downstream processing
    """
    # Validate projections dimensions
    if projections.dim() != 4:
        raise ValueError("Projections must be a 4D tensor (B, P, height, width/2+1)")
    
    B, P, proj_boxsize, proj_boxsize_half = projections.shape
    
    # Enforce square, even dimensions
    if proj_boxsize % 2 != 0:
        raise ValueError(f"Projection boxsize ({proj_boxsize}) must be even. Only even dimensions are supported.")
    if proj_boxsize != (proj_boxsize_half - 1) * 2:
        raise ValueError(f"Projection shape mismatch: expected boxsize {proj_boxsize} to match 2*(boxsize_half-1) = {(proj_boxsize_half - 1) * 2}")
    
    # Validate optional weights
    if weights is not None:
        if weights.shape != projections.shape:
            raise ValueError(f"Weights shape {weights.shape} must match projections shape {projections.shape}")
        if not weights.is_floating_point():
            raise ValueError("Weights must be real-valued (floating point)")
    
    return _Backproject2D.apply(projections, weights, rotations, shifts, interpolation, oversampling, fourier_radius_cutoff)

def project_3d_to_2d_forw(
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
    
    return _Project3DTo2D.apply(reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff)

def project_3d_to_2d_back(
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
    grad_reconstruction, _, _ = torch.ops.torch_projectors.project_3d_to_2d_back(
        projections, reconstruction, rotations, shifts, interpolation, oversampling, None
    )
    
    return grad_reconstruction

def backproject_2d_to_3d_forw(
    projections: torch.Tensor,
    rotations: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    shifts: Optional[torch.Tensor] = None,
    interpolation: str = 'linear',
    oversampling: float = 1.0,
    fourier_radius_cutoff: Optional[float] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Back-project 2D projections into 3D reconstructions (adjoint/transpose operation)
    
    This function accumulates 2D projection data (and optional weights) into 3D reconstructions using the Central
    Slice Theorem. It is the mathematical adjoint/transpose of 3D->2D forward projection.
    
    Args:
        projections: 4D complex tensor [B, P, height, width/2+1] - 2D projections in FFTW format
        rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices
        weights: Optional 4D real tensor [B, P, height, width/2+1] - weights (e.g., CTF^2)
        shifts: Optional 3D real tensor [B_shift, P, 2] - 2D translation shifts
        interpolation: 'linear' (trilinear) or 'cubic' (tricubic) interpolation
        oversampling: Coordinate scaling factor (>1 for oversampling)
        fourier_radius_cutoff: Optional frequency cutoff for low-pass filtering
        
    Returns:
        Tuple of (data_reconstruction, weight_reconstruction):
        - data_reconstruction: 4D complex tensor [B, D, H, W/2+1] - accumulated projection data
        - weight_reconstruction: 4D real tensor [B, D, H, W/2+1] - accumulated weights
          (empty tensor if weights=None)
        
    Note:
        - This is the adjoint operation of project_3d_to_2d_forw()
        - Uses conjugate phase shifts for proper mathematical adjoint
        - Supports batch broadcasting: B_rot and B_shift can be 1 or match B
        - Output volume is cubic with depth=height=width based on projection dimensions
        - Weight accumulation enables Wiener-like filtering in downstream processing
    """
    # Validate projections dimensions
    if projections.dim() != 4:
        raise ValueError("Projections must be a 4D tensor (B, P, height, width/2+1)")
    
    B, P, proj_boxsize, proj_boxsize_half = projections.shape
    
    # Enforce square, even dimensions
    if proj_boxsize % 2 != 0:
        raise ValueError(f"Projection boxsize ({proj_boxsize}) must be even. Only even dimensions are supported.")
    if proj_boxsize != (proj_boxsize_half - 1) * 2:
        raise ValueError(f"Projection shape mismatch: expected boxsize {proj_boxsize} to match 2*(boxsize_half-1) = {(proj_boxsize_half - 1) * 2}")
    
    # Validate optional weights
    if weights is not None:
        if weights.shape != projections.shape:
            raise ValueError(f"Weights shape {weights.shape} must match projections shape {projections.shape}")
        if not weights.is_floating_point():
            raise ValueError("Weights must be real-valued (floating point)")
    
    return _Backproject2DTo3D.apply(projections, weights, rotations, shifts, interpolation, oversampling, fourier_radius_cutoff)

def backproject_2d_to_3d_back(
    reconstruction: torch.Tensor,
    projections: torch.Tensor,
    rotations: torch.Tensor,
    shifts: Optional[torch.Tensor] = None,
    interpolation: str = 'linear',
    oversampling: float = 1.0
) -> torch.Tensor:
    """
    Backward project 3D reconstruction to 2D projections (returns only projection gradients)

    This function computes only the 2D projection gradients from the unified backward
    projection operation, discarding rotation and shift gradients for convenience.

    Args:
        reconstruction: 4D complex tensor [B, D, H, W/2+1] - 3D reconstruction gradients
        projections: 4D complex tensor [B, P, H, W/2+1] - 2D projections
        rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices
        shifts: Optional 3D real tensor [B_shift, P, 2] - 2D translation shifts
        interpolation: 'linear' (trilinear) or 'cubic' (tricubic) interpolation
        oversampling: Coordinate scaling factor (must match forward pass)

    Returns:
        4D complex tensor [B, P, H, W/2+1] - gradients w.r.t. 2D projections
    """
    # Call the unified backward function and return only projection gradients
    grad_projections, _, _ = torch.ops.torch_projectors.backproject_2d_to_3d_back(
        reconstruction, projections, rotations, shifts, interpolation, oversampling, None
    )

    return grad_projections

def project_3d_forw(
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts: Optional[torch.Tensor] = None,
    output_shape: Optional[tuple[int, ...]] = None,
    interpolation: str = 'linear',
    oversampling: float = 1.0,
    fourier_radius_cutoff: Optional[float] = None
) -> torch.Tensor:
    """
    Forward project 3D Fourier volume to 3D projections using full 3D rotation.

    This function projects 4D Fourier-space reconstructions [B, D, H, W/2+1] to 3D projections
    [B, P, D_out, H_out, W_out/2+1] by sampling the 3D volume at rotated coordinates.
    Unlike 3D->2D projection which uses the central slice theorem, this performs full 3D
    rotation for each output voxel.

    Args:
        reconstruction: 4D complex tensor [B, D, H, W/2+1] - 3D Fourier volume in FFTW format
        rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices
        shifts: Optional 3D real tensor [B_shift, P, 3] - 3D translation shifts
        output_shape: Tuple (D_out, H_out, W_out) for output projection size. Defaults to (D, H, W)
        interpolation: 'linear' (trilinear) or 'cubic' (tricubic) interpolation
        oversampling: Coordinate scaling factor (>1 for oversampling)
        fourier_radius_cutoff: Optional frequency cutoff for low-pass filtering

    Returns:
        5D complex tensor [B, P, D_out, H_out, W_out/2+1] - the 3D projections

    Note:
        - Uses full 3D rotation (not limited to central slice)
        - Supports batch broadcasting: B_rot and B_shift can be 1 or match B
        - 3D Friedel symmetry is automatically handled for real-valued reconstructions
    """
    # Validate reconstruction dimensions
    if reconstruction.dim() != 4:
        raise ValueError("Reconstruction must be a 4D tensor (B, D, H, W/2+1)")

    B, D, boxsize, boxsize_half = reconstruction.shape

    # Enforce cubic, even dimensions
    if boxsize % 2 != 0:
        raise ValueError(f"Boxsize ({boxsize}) must be even. Only even dimensions are supported.")
    if boxsize != (boxsize_half - 1) * 2:
        raise ValueError(f"Reconstruction shape mismatch: expected boxsize {boxsize} to match 2*(boxsize_half-1) = {(boxsize_half - 1) * 2}")
    if D != boxsize:
        raise ValueError(f"Reconstruction must be cubic: depth {D} must equal boxsize {boxsize}")

    # Set default output shape to cubic
    if output_shape is None:
        output_shape = (D, boxsize, boxsize)
    else:
        # Validate output shape is cubic and even
        if len(output_shape) != 3:
            raise ValueError(f"Output shape {output_shape} must be 3D (D, H, W)")
        if output_shape[0] != output_shape[1] or output_shape[1] != output_shape[2]:
            raise ValueError(f"Output shape {output_shape} must be cubic (D == H == W)")
        if output_shape[0] % 2 != 0:
            raise ValueError(f"Output boxsize {output_shape[0]} must be even")

    return _Project3D.apply(reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff)

def project_3d_back(
    projections: torch.Tensor,
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts: Optional[torch.Tensor] = None,
    interpolation: str = 'linear',
    oversampling: float = 1.0
) -> torch.Tensor:
    """
    Backward project 3D projections to 3D reconstruction (returns only reconstruction gradients)

    This function computes only the 3D reconstruction gradients from the unified backward
    projection operation, discarding rotation and shift gradients for convenience.

    Args:
        projections: 5D complex tensor [B, P, D, H, W/2+1] - 3D projections
        reconstruction: 4D complex tensor [B, D, H, W/2+1] - 3D Fourier volume
        rotations: 4D real tensor [B_rot, P, 3, 3] - 3x3 rotation matrices
        shifts: Optional 3D real tensor [B_shift, P, 3] - 3D translation shifts
        interpolation: 'linear' (trilinear) or 'cubic' (tricubic) interpolation
        oversampling: Coordinate scaling factor (must match forward pass)

    Returns:
        4D complex tensor [B, D, H, W/2+1] - gradients w.r.t. 3D reconstruction
    """
    # Call the unified backward function and return only reconstruction gradients
    grad_reconstruction, _, _ = torch.ops.torch_projectors.project_3d_back(
        projections, reconstruction, rotations, shifts, interpolation, oversampling, None
    )

    return grad_reconstruction

# --- Autograd Registration ---

class _Project2D(torch.autograd.Function):
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
        
        projection = torch.ops.torch_projectors.project_2d_forw(reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff)
        
        ctx.save_for_backward(reconstruction, rotations, shifts)
        ctx.interpolation = interpolation
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        return projection

    @staticmethod
    def backward(ctx, grad_output):
        reconstruction, rotations, shifts = ctx.saved_tensors
        
        grad_reconstruction, grad_rotations, grad_shifts = torch.ops.torch_projectors.project_2d_back(
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

class _Project3DTo2D(torch.autograd.Function):
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
        
        projection = torch.ops.torch_projectors.project_3d_to_2d_forw(reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff)
        
        ctx.save_for_backward(reconstruction, rotations, shifts)
        ctx.interpolation = interpolation
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        return projection

    @staticmethod
    def backward(ctx, grad_output):
        reconstruction, rotations, shifts = ctx.saved_tensors
        
        grad_reconstruction, grad_rotations, grad_shifts = torch.ops.torch_projectors.project_3d_to_2d_back(
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

class _Backproject2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projections, weights, rotations, shifts, interpolation, oversampling, fourier_radius_cutoff):
        # Unsqueeze if single projection/pose set
        if projections.dim() == 3:
            projections = projections.unsqueeze(0)
        if rotations.dim() == 2:
            rotations = rotations.unsqueeze(0)
        if rotations.dim() == 3:
            rotations = rotations.unsqueeze(0)
        if shifts is not None:
            if shifts.dim() == 1:
                shifts = shifts.unsqueeze(0)
            if shifts.dim() == 2:
                shifts = shifts.unsqueeze(0)
        if weights is not None:
            if weights.dim() == 3:
                weights = weights.unsqueeze(0)

        # Let C++ handle batch size mismatches (1 vs N)
        if rotations.size(0) != projections.size(0) and rotations.size(0) != 1:
            raise ValueError("Batch size of rotations must be 1 or match projections")
        
        data_reconstruction, weight_reconstruction = torch.ops.torch_projectors.backproject_2d_forw(
            projections, weights, rotations, shifts, interpolation, oversampling, fourier_radius_cutoff)
        
        ctx.save_for_backward(projections, weights, rotations, shifts)
        ctx.interpolation = interpolation
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        return data_reconstruction, weight_reconstruction

    @staticmethod
    def backward(ctx, grad_data_rec, grad_weight_rec):
        projections, weights, rotations, shifts = ctx.saved_tensors
        
        # Handle case where grad_weight_rec is an empty tensor (when weights weren't provided)
        if grad_weight_rec is not None and grad_weight_rec.numel() == 0:
            grad_weight_rec = None
        
        grad_projections, grad_weights, grad_rotations, grad_shifts = torch.ops.torch_projectors.backproject_2d_back(
            grad_data_rec.contiguous(),
            grad_weight_rec.contiguous() if grad_weight_rec is not None else None,
            projections,
            weights,
            rotations,
            shifts,
            ctx.interpolation,
            ctx.oversampling,
            ctx.fourier_radius_cutoff
        )
        
        # If inputs were None in the forward pass, return None for their gradients
        if weights is None:
            grad_weights = None
        if shifts is None:
            grad_shifts = None
        
        return grad_projections, grad_weights, grad_rotations, grad_shifts, None, None, None

class _Backproject2DTo3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projections, weights, rotations, shifts, interpolation, oversampling, fourier_radius_cutoff):
        # Unsqueeze if single projection/pose set
        if projections.dim() == 3:
            projections = projections.unsqueeze(0)
        if rotations.dim() == 2:
            rotations = rotations.unsqueeze(0)
        if rotations.dim() == 3:
            rotations = rotations.unsqueeze(0)
        if shifts is not None:
            if shifts.dim() == 1:
                shifts = shifts.unsqueeze(0)
            if shifts.dim() == 2:
                shifts = shifts.unsqueeze(0)
        if weights is not None:
            if weights.dim() == 3:
                weights = weights.unsqueeze(0)

        # Let C++ handle batch size mismatches (1 vs N)
        if rotations.size(0) != projections.size(0) and rotations.size(0) != 1:
            raise ValueError("Batch size of rotations must be 1 or match projections")

        data_reconstruction, weight_reconstruction = torch.ops.torch_projectors.backproject_2d_to_3d_forw(
            projections, weights, rotations, shifts, interpolation, oversampling, fourier_radius_cutoff)

        ctx.save_for_backward(projections, weights, rotations, shifts)
        ctx.interpolation = interpolation
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        return data_reconstruction, weight_reconstruction

    @staticmethod
    def backward(ctx, grad_data_rec, grad_weight_rec):
        projections, weights, rotations, shifts = ctx.saved_tensors

        # Handle case where grad_weight_rec is an empty tensor (when weights weren't provided)
        if grad_weight_rec is not None and grad_weight_rec.numel() == 0:
            grad_weight_rec = None

        grad_projections, grad_weights, grad_rotations, grad_shifts = torch.ops.torch_projectors.backproject_2d_to_3d_back(
            grad_data_rec.contiguous(),
            grad_weight_rec.contiguous() if grad_weight_rec is not None else None,
            projections,
            weights,
            rotations,
            shifts,
            ctx.interpolation,
            ctx.oversampling,
            ctx.fourier_radius_cutoff
        )

        # If inputs were None in the forward pass, return None for their gradients
        if weights is None:
            grad_weights = None
        if shifts is None:
            grad_shifts = None

        return grad_projections, grad_weights, grad_rotations, grad_shifts, None, None, None

class _Project3D(torch.autograd.Function):
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

        projection = torch.ops.torch_projectors.project_3d_forw(reconstruction, rotations, shifts, output_shape, interpolation, oversampling, fourier_radius_cutoff)

        ctx.save_for_backward(reconstruction, rotations, shifts)
        ctx.interpolation = interpolation
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        return projection

    @staticmethod
    def backward(ctx, grad_output):
        reconstruction, rotations, shifts = ctx.saved_tensors

        grad_reconstruction, grad_rotations, grad_shifts = torch.ops.torch_projectors.project_3d_back(
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

def _project_2d_back_backward(ctx, grad_output):
    # No-op placeholder
    return None

def _project_3d_to_2d_back_backward(ctx, grad_output):
    # No-op placeholder
    return None

def _backproject_2d_forw_backward(ctx, grad_output):
    # No-op placeholder for backproject_2d_forw
    return None

def _backproject_2d_back_backward(ctx, grad_output):
    # No-op placeholder for backproject_2d_back
    return None

def _backproject_2d_to_3d_forw_backward(ctx, grad_output):
    # No-op placeholder for backproject_2d_to_3d_forw
    return None

def _backproject_2d_to_3d_back_backward(ctx, grad_output):
    # No-op placeholder for backproject_2d_to_3d_back
    return None

torch.library.register_autograd(
    "torch_projectors::project_2d_back", 
    _project_2d_back_backward,
    setup_context=lambda ctx, inputs, output: None
)

torch.library.register_autograd(
    "torch_projectors::backproject_2d_forw", 
    _backproject_2d_forw_backward,
    setup_context=lambda ctx, inputs, output: None
)

torch.library.register_autograd(
    "torch_projectors::backproject_2d_back", 
    _backproject_2d_back_backward,
    setup_context=lambda ctx, inputs, output: None
)

torch.library.register_autograd(
    "torch_projectors::project_3d_to_2d_back", 
    _project_3d_to_2d_back_backward,
    setup_context=lambda ctx, inputs, output: None
)

torch.library.register_autograd(
    "torch_projectors::backproject_2d_to_3d_forw", 
    _backproject_2d_to_3d_forw_backward,
    setup_context=lambda ctx, inputs, output: None
)

torch.library.register_autograd(
    "torch_projectors::backproject_2d_to_3d_back",
    _backproject_2d_to_3d_back_backward,
    setup_context=lambda ctx, inputs, output: None
)

def _project_3d_back_backward(ctx, grad_output):
    # No-op placeholder for project_3d_back
    return None

torch.library.register_autograd(
    "torch_projectors::project_3d_back",
    _project_3d_back_backward,
    setup_context=lambda ctx, inputs, output: None
) 