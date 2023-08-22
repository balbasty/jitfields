from jitfields.resize import resize
from jitfields.utils import identity_grid
from .utils import get_test_devices, init_device
import torch
import pytest

bounds = ['dct1', 'dct2', 'dft']
orders = list(range(2, 8))
anchors = ['center', 'edge', 'first']
devices = get_test_devices()
dims = [1, 2, 3]
shape1 = 32
dtype = torch.float64


def make_data(shape, device, dtype):
    batch = 2
    grid = identity_grid(shape, device=device, dtype=dtype)
    vol = torch.randn([batch, *shape], device=device, dtype=dtype)
    return vol, grid


@pytest.mark.parametrize("device", devices)
def test_resize(device):
    device = init_device(device)

    inp = [[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]]

    out = [[.00, .00, .00, .00, .00],
           [.00, .25, .50, .25, .00],
           [.00, .50, 1.0, .50, .00],
           [.00, .25, .50, .25, .00],
           [.00, .00, .00, .00, .00]]

    inp = torch.as_tensor(inp, device=device, dtype=torch.float32)
    out = torch.as_tensor(out, device=device, dtype=torch.float32)
    pred = resize(inp, order=1, anchor='c', shape=[5, 5], ndim=2)
    assert torch.allclose(out.to(pred), pred)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("order", orders)
@pytest.mark.parametrize("anchor", anchors)
def test_resize_same(device, dim, bound, order, anchor):
    print(f'pull_{dim}d({order}, {bound}) on {device}')
    device = init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    pred = resize(vol, 1, shape, dim, anchor, order, bound, True)
    assert torch.allclose(pred, vol), f"\n{vol}\n{pred}\n"
