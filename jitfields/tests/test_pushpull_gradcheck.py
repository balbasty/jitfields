import torch
from torch.autograd import gradcheck
from jitfields.pushpull import pull, push, count, grad
from jitfields.utils import add_identity_grid_
from jitfields.bindings.common.bounds import cnames as boundnames
from .utils import test_devices, init_device
import inspect
import pytest
import os
os.environ['CUDA_NUM_THREADS'] = '64'

# global parameters
dtype = torch.double        # data type (double advised to check gradients)
shape1 = 3                  # size along each dimension
extrapolate = True

# parameters
bounds = [boundnames[i].lower() for i in range(7)]
orders = list(range(8))[:3]
devices = test_devices()
dims = [1, 2, 3]


if hasattr(torch, 'use_deterministic_algorithms'):
    torch.use_deterministic_algorithms(True)
kwargs = dict(
    rtol=1.,
    raise_exception=True,
    check_grad_dtypes=True,
)
if 'check_undefined_grad' in inspect.signature(gradcheck).parameters:
    kwargs['check_undefined_grad'] = False
if 'nondet_tol' in inspect.signature(gradcheck).parameters:
    kwargs['nondet_tol'] = float('inf')


def make_data(shape, device, dtype):
    batch, channel = 1, 1
    grid = torch.randn([batch, *shape, len(shape)], device=device, dtype=dtype)
    grid = add_identity_grid_(grid)
    vol = torch.randn([batch, *shape, channel], device=device, dtype=dtype)
    return vol, grid


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
def test_gradcheck_grad(device, dim, bound, interpolation):
    print(f'grad_{dim}d({interpolation}, {bound}) on {device}')
    device = init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    vol.requires_grad = True
    grid.requires_grad = True
    assert gradcheck(grad, (vol, grid, interpolation, bound, extrapolate),
                     **kwargs)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
def test_gradcheck_pull(device, dim, bound, interpolation):
    print(f'pull_{dim}d({interpolation}, {bound}) on {device}')
    device = init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    vol.requires_grad = True
    grid.requires_grad = True
    assert gradcheck(pull, (vol, grid, interpolation, bound, extrapolate),
                     **kwargs)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
def test_gradcheck_push(device, dim, bound, interpolation):
    print(f'push_{dim}d({interpolation}, {bound}) on {device}')
    device = init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    vol.requires_grad = True
    grid.requires_grad = True
    assert gradcheck(push, (vol, grid, shape, interpolation, bound, extrapolate),
                     **kwargs)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
def test_gradcheck_count(device, dim, bound, interpolation):
    print(f'count_{dim}d({interpolation}, {bound}) on {device}')
    device = init_device(device)
    shape = (shape1,) * dim
    _, grid = make_data(shape, device, dtype)
    grid.requires_grad = True
    assert gradcheck(count, (grid, shape, interpolation, bound, extrapolate),
                     **kwargs)
