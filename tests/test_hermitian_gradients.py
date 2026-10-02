"""Check native Fourier operators through physically valid real-space losses.

Every native input is an RFFT of a real array, and every loss is squared error
following IRFFT. The loss neither excludes boundary bins nor manually weights
redundant Fourier samples. Finite differences call the same native forward.

CPU kernels use float64; CUDA and MPS kernels use float32. FFTs and scalar
reductions run on CPU so MPS does not require complex FFT support. Casting on
CPU before copying to the accelerator also keeps float64 out of MPS backward.
"""

import math

import pytest
import torch
import torch_projectors as tp


MODES = (
    "project_2d",
    "project_3d_to_2d",
    "backproject_2d",
    "backproject_2d_to_3d",
)


def _rotation(angles, dimension):
    def plane(angle, axis):
        cosine, sine = angle.cos(), angle.sin()
        zero, one = angle * 0, angle * 0 + 1
        if dimension == 2:
            return torch.stack((cosine, -sine, sine, cosine)).reshape(2, 2)
        entries = (
            (one, zero, zero, zero, cosine, -sine, zero, sine, cosine),
            (cosine, zero, sine, zero, one, zero, -sine, zero, cosine),
            (cosine, -sine, zero, sine, cosine, zero, zero, zero, one),
        )[axis]
        return torch.stack(entries).reshape(3, 3)

    if dimension == 2:
        return plane(angles[0], 0)
    return plane(angles[2], 2) @ plane(angles[1], 1) @ plane(angles[0], 0)


@pytest.fixture(params=("cpu", "cuda", "mps"))
def backend(request):
    name = request.param
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    return torch.device(name)


def _problem(
    mode,
    interpolation,
    spectrum,
    backend,
    oversampling=1,
    batch_size=1,
    num_poses=1,
):
    output_size = 12
    is_backprojection = mode.startswith("back")
    dimension = 3 if "3d" in mode else 2
    input_dimension = 2 if is_backprojection else dimension
    output_dimension = dimension if is_backprojection else 2
    input_size = output_size if is_backprojection else output_size * oversampling
    batch_shape = (batch_size, num_poses) if is_backprojection else (batch_size,)
    shape = batch_shape + (input_size,) * input_dimension
    generator = torch.Generator().manual_seed(1729 + dimension)
    source = torch.randn(shape, generator=generator, dtype=torch.float64)
    direction = source + 0.3 * torch.randn(
        shape, generator=generator, dtype=torch.float64
    )

    if spectrum == "boundary":
        # Constant real-space x gives k_x=0, including BOTH conjugate halves.
        source = source.mean(-1, keepdim=True).expand(shape).contiguous()
        direction = direction.mean(-1, keepdim=True).expand(shape).contiguous()
        source *= math.sqrt(input_size)
        direction *= math.sqrt(input_size)
    elif spectrum == "dc":
        source = torch.ones(shape, dtype=torch.float64)
        direction = source * 0.7
    elif spectrum == "nyquist":
        # Alternating pixels create self-conjugate Nyquist axes and corners.
        # Keep DC so every mode has a nonzero input derivative.
        coordinates = torch.meshgrid(
            *([torch.arange(input_size)] * input_dimension), indexing="ij"
        )
        source = torch.ones(shape, dtype=torch.float64)
        for axis, coordinate in enumerate(coordinates):
            source = source + (axis + 1) * torch.where(coordinate % 2 == 0, 1.0, -1.0)
        source = source + torch.where(sum(coordinates) % 2 == 0, 1.0, -1.0)
        direction = source * 0.7

    # Non-axis-aligned orientations selected away from interpolation knots at
    # both finite-difference step sizes, including the oversampled fixture.
    angles = torch.tensor(
        [0.237] if dimension == 2 else [0.237, -0.329, 0.483], dtype=torch.float64
    )
    shifts = torch.tensor([0.217, -0.361], dtype=torch.float64)
    dtype = torch.float64 if backend.type == "cpu" else torch.float32
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    function = getattr(tp, mode + "_forw")

    def predict(rotation_angles, translation, input_step):
        real_input = source + input_step * direction
        fourier_input = torch.fft.rfftn(
            real_input, dim=tuple(range(-input_dimension, 0))
        )
        # Shared rotation/shift batches exercise native broadcast reductions.
        # Distinct poses ensure this is more than repeated identical work.
        rotations = torch.stack(
            [
                _rotation(rotation_angles + 0.071 * pose, dimension)
                for pose in range(num_poses)
            ]
        )[None]
        translations = torch.stack(
            [translation + 0.053 * pose for pose in range(num_poses)]
        )[None]
        kwargs = {} if is_backprojection else {"output_shape": (output_size,) * 2}
        output = function(
            fourier_input.to(dtype=complex_dtype).to(backend),
            rotations.to(dtype=dtype).to(backend),
            shifts=translations.to(dtype=dtype).to(backend),
            interpolation=interpolation,
            oversampling=oversampling,
            **kwargs,
        )
        if is_backprojection:
            output = output[0]
        return torch.fft.irfftn(
            output.cpu(),
            s=(output.shape[-2],) * output_dimension,
            dim=tuple(range(-output_dimension, 0)),
        ).double()

    # The native forward generates a nearby, fixed target for every derivative.
    target = predict(angles + 0.13, shifts + 0.19, 0.17).detach()

    def loss(parameter, derivative):
        prediction = predict(
            parameter if derivative == "rotation" else angles,
            parameter if derivative == "shift" else shifts,
            parameter[0] if derivative == "input" else 0.0,
        )
        return 0.5 * (prediction - target).square().sum()

    initial = {
        "rotation": angles,
        "shift": shifts,
        "input": torch.zeros(1, dtype=torch.float64),
    }
    return loss, initial


def _check_derivative(loss, initial, derivative, backend, label):
    parameter = initial[derivative].clone().requires_grad_()
    (analytic,) = torch.autograd.grad(loss(parameter, derivative), parameter)
    # Two steps guard against accidental agreement at one finite-difference
    # step. Float64 also provides a strict interpolation derivative check.
    steps = (1e-6, 2e-6) if backend.type == "cpu" else (2e-4, 4e-4)
    tolerance = 2e-6 if backend.type == "cpu" else 5e-3
    estimates = []
    with torch.no_grad():
        for step in steps:
            numerical = torch.empty_like(parameter)
            for index in range(parameter.numel()):
                delta = torch.zeros_like(parameter)
                delta[index] = step
                numerical[index] = (
                    loss(parameter + delta, derivative)
                    - loss(parameter - delta, derivative)
                ) / (2 * step)
            estimates.append(numerical)

    for numerical in estimates:
        scale = max(float(analytic.norm()), float(numerical.norm()), 1e-10)
        relative_error = float((analytic - numerical).norm()) / scale
        assert relative_error < tolerance, (
            f"{label}/{derivative}/{backend}: "
            f"analytic={analytic.tolist()}, finite_difference={numerical.tolist()}, "
            f"relative_error={relative_error:.8g}, tolerance={tolerance}"
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("interpolation", ("linear", "cubic"))
@pytest.mark.parametrize("spectrum", ("full", "boundary"))
@pytest.mark.parametrize("derivative", ("rotation", "shift", "input"))
def test_real_space_hermitian_gradients(
    backend, mode, interpolation, spectrum, derivative
):
    loss, initial = _problem(mode, interpolation, spectrum, backend)
    _check_derivative(
        loss, initial, derivative, backend, f"{mode}/{interpolation}/{spectrum}"
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("interpolation", ("linear", "cubic"))
@pytest.mark.parametrize("derivative", ("rotation", "shift", "input"))
def test_oversampled_broadcast_hermitian_gradients(
    backend, mode, interpolation, derivative
):
    """Two independent images and two poses share pose batches at 2x sampling."""
    loss, initial = _problem(
        mode, interpolation, "full", backend,
        oversampling=2, batch_size=2, num_poses=2,
    )
    _check_derivative(
        loss, initial, derivative, backend,
        f"{mode}/{interpolation}/oversampling2/broadcast",
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("interpolation", ("linear", "cubic"))
@pytest.mark.parametrize("spectrum", ("dc", "nyquist"))
def test_self_conjugate_real_input_gradients(backend, mode, interpolation, spectrum):
    """Special bins remain valid real-input directions, never free complex bins."""
    loss, initial = _problem(mode, interpolation, spectrum, backend)
    _check_derivative(
        loss, initial, "input", backend, f"{mode}/{interpolation}/{spectrum}"
    )
