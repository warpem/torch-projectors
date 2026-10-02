"""Regressions for weight/pose derivatives through a normalized real-space reconstruction."""
import pytest
import torch
import torch_projectors as tp


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("denominator_only", [False, True])
def test_backprojection_normalized_real_loss_weight_gradients(
    device_type, ndim, interpolation, denominator_only
):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS device unavailable")
    dtype = torch.float64 if device_type == "cpu" else torch.float32
    generator = torch.Generator().manual_seed(54 + ndim)
    n = 10
    projections = torch.fft.rfft2(torch.randn(1, 2, n, n, dtype=dtype, generator=generator))
    try:
        projections = projections.to(device_type)
    except (RuntimeError, TypeError) as exc:
        if device_type == "mps" and "complex" in str(exc).lower():
            pytest.skip("Installed PyTorch cannot allocate complex MPS tensors")
        raise
    weights = torch.rand(projections.shape, dtype=dtype, generator=generator) + .3
    shifts = torch.tensor([[[.24, -.17], [.07, .15]]], dtype=dtype, device=device_type)
    angles = torch.tensor([.231, -.317] if ndim == 2 else [.231, -.317, .123], dtype=dtype)
    target = torch.randn((1,) + (n,) * ndim, dtype=dtype, generator=generator) * .02
    operation = tp.backproject_2d_forw if ndim == 2 else tp.backproject_2d_to_3d_forw

    def rotations(a):
        if ndim == 2:
            c, s = torch.cos(a), torch.sin(a)
            return torch.stack((c, -s, s, c), -1).reshape(1, 2, 2, 2)
        zero = torch.zeros_like(a[0])
        x, y, z = a
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
        rotation = torch.matrix_exp(skew)
        tilt = torch.tensor([[1., 0., 0.], [0., .8, -.6], [0., .6, .8]], dtype=dtype)
        return torch.stack((rotation, tilt @ rotation))[None]

    def reconstruct(a, w):
        return operation(
            projections, rotations(a).to(device_type), weights=w.to(device_type),
            shifts=shifts, interpolation=interpolation, fourier_radius_cutoff=3.5,
        )

    fixed_data, _ = reconstruct(angles, weights)
    fixed_data = fixed_data.detach().cpu()

    def loss(a, w):
        data, denominator = reconstruct(a, w)
        # This loss observes real voxels, not a sum over redundant Fourier entries.
        data = fixed_data if denominator_only else data.cpu()
        image = torch.fft.irfftn(
            data / (denominator.cpu() + 2), s=(n,) * ndim,
            dim=tuple(range(-ndim, 0)),
        )
        return .5 * ((image.double() - target.double()) ** 2).sum()

    differentiable_angles = angles.clone().requires_grad_()
    differentiable_weights = weights.clone().requires_grad_()
    angle_gradient, weight_gradient = torch.autograd.grad(
        loss(differentiable_angles, differentiable_weights),
        (differentiable_angles, differentiable_weights),
    )
    eps = 1e-5 if device_type == "cpu" else 5e-4
    finite_angle_gradient = torch.zeros_like(angles)
    for index in range(angles.numel()):
        delta = torch.zeros_like(angles)
        delta[index] = eps
        finite_angle_gradient[index] = (loss(angles + delta, weights) - loss(angles - delta, weights)) / (2 * eps)
    angle_error = (angle_gradient - finite_angle_gradient).norm() / finite_angle_gradient.norm()
    assert finite_angle_gradient.norm() > 1e-5  # Denominator branch is measurably pose dependent.
    assert angle_error < (2e-5 if device_type == "cpu" else 1e-2), angle_error.item()

    # The denominator must still contribute to pose when input weights are constants.
    constant_weight_angles = angles.clone().requires_grad_()
    (constant_weight_angle_gradient,) = torch.autograd.grad(
        loss(constant_weight_angles, weights), (constant_weight_angles,)
    )
    torch.testing.assert_close(constant_weight_angle_gradient, angle_gradient, rtol=2e-5, atol=1e-7)

    analytic_weight_directions, finite_weight_directions = [], []
    for _ in range(4):
        direction = torch.randn(weights.shape, dtype=dtype, generator=generator)
        direction /= direction.norm()
        analytic_weight_directions.append((weight_gradient * direction).sum())
        finite_weight_directions.append(
            (loss(angles, weights + eps * direction) - loss(angles, weights - eps * direction)) / (2 * eps)
        )
    analytic_weight_directions = torch.stack(analytic_weight_directions)
    finite_weight_directions = torch.stack(finite_weight_directions)
    weight_error = (analytic_weight_directions - finite_weight_directions).norm() / finite_weight_directions.norm()
    assert weight_error < (2e-5 if device_type == "cpu" else 1e-2), weight_error.item()
