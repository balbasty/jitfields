import torch
from torch.autograd import gradcheck
from jitfields.resize import resize, restrict
from jitfields.bindings.common.bounds import cnames as boundnames
from .utils import test_devices, init_device
import inspect
import pytest
import os
os.environ['CUDA_NUM_THREADS'] = '64'

# global parameters
dtype = torch.double        # data type (double advised to check gradients)
shape1 = 5                  # size along each dimension
extrapolate = True
prefilter = False

# parameters
bounds = [boundnames[i].lower() for i in range(7)][3:4]
orders = list(range(8))[:4]
anchors = ['center', 'edge', 'first'][:1]
factors = [2, 3]
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
    batch = 1
    vol = torch.randn([batch, *shape], device=device, dtype=dtype)
    return vol


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
@pytest.mark.parametrize("factor", factors)
@pytest.mark.parametrize("anchor", anchors)
def test_gradcheck_resize(device, dim, bound, interpolation, factor, anchor):
    print(f'resize_{dim}d({factor}, {interpolation}, {bound}, {anchor}) on {device}')
    device = init_device(device)
    shape = (shape1,) * dim
    vol = make_data(shape, device, dtype)
    vol.requires_grad = True
    assert gradcheck(resize, (vol, factor, None, dim, anchor, interpolation,
                              bound, prefilter), **kwargs)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
@pytest.mark.parametrize("factor", factors)
@pytest.mark.parametrize("anchor", anchors)
def test_adjoint_resize(device, dim, bound, interpolation, factor, anchor):
    print(f'resize_{dim}d({factor}, {interpolation}, {bound}, {anchor}) on {device}')
    device = init_device(device)
    torch.random.manual_seed(0)
    shapeinp = (16,) * dim
    shapeout = (16*factor,) * dim
    u = torch.randn(shapeinp, device=device, dtype=dtype)
    Au = resize(u, None, shapeout, dim, anchor, interpolation, bound, prefilter)
    v = torch.randn_like(Au)
    Atv = restrict(v, None, shapeinp, dim, anchor, interpolation, bound, True)

    vAu = v.flatten().dot(Au.flatten())
    uAtv = u.flatten().dot(Atv.flatten())
    print(f"<v, Au> = {vAu.item()}, <u, A'v> = {uAtv.item()}")
    assert torch.allclose(vAu, uAtv), f"<v, Au> = {vAu.item()}, <u, A'v> = {uAtv.item()}"
