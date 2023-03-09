import torch
from torch.autograd import gradcheck
from jitfields.pushpull import pull, push, count, grad
from jitfields.utils import identity_grid
from jitfields.bindings.common.bounds import cnames as boundnames
from .utils import test_devices, init_device
import inspect
import pytest
import os
os.environ['CUDA_NUM_THREADS'] = '64'

# global parameters
dtype = torch.double        # data type (double advised to check gradients)
shape1 = 32                 # size along each dimension
extrapolate = True

# parameters
bounds = ['dct1', 'dct2', 'dft']
orders = list(range(8))
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
    grid = identity_grid(shape, device=device, dtype=dtype)
    vol = torch.randn([batch, *shape, channel], device=device, dtype=dtype)
    return vol, grid


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("order", orders)
def test_pull_same(device, dim, bound, order):
    print(f'pull_{dim}d({order}, {bound}) on {device}')
    device = init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    pred = pull(vol, grid, order, bound, True, True)
    assert torch.allclose(pred, vol), f"\n{vol}\n{pred}\n"
