from jitfields.regularisers import grid_kernel, grid_vel2mom
from .ref_kernels import kernels2, kernels3
from .utils import test_devices, init_device
import torch
import pytest


devices = test_devices()
dims = [2, 3]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
def test_kernel(device, dim):

    device = init_device(device)
    backend = dict(device=device, dtype=torch.float32)

    kernel_absolute = torch.zeros([3]*dim, **backend)
    kernel_absolute[(1,)*dim] = 1

    if dim == 2:
        kernels = kernels2
    elif dim == 3:
        kernels = kernels3
    else:
        assert False, f"Test not implemented for dim {dim}"

    pred_absolute = grid_kernel([3]*dim, absolute=1, **backend)[..., 0]
    kernel_absolute = torch.as_tensor(kernel_absolute, **backend)
    assert torch.allclose(pred_absolute, kernel_absolute), f"{pred_absolute}\n{kernel_absolute}"

    pred_membrane = grid_kernel([3]*dim, membrane=1, **backend)[..., 0]
    kernel_membrane = torch.as_tensor(kernels.membrane, **backend)
    assert torch.allclose(pred_membrane, kernel_membrane), f"{pred_membrane}\n{kernel_membrane}"

    pred_bending = grid_kernel([5]*dim, bending=1, **backend)[..., 0]
    kernel_bending = torch.as_tensor(kernels.bending, **backend)
    assert torch.allclose(pred_bending, kernel_bending), f"{pred_bending}\n{kernel_bending}"

    pred_shears = grid_kernel([3]*dim, shears=1, **backend)
    kernel_shears = torch.as_tensor(kernels.shears, **backend).movedim(0, -1).movedim(0, -1)
    assert torch.allclose(pred_shears, kernel_shears), f"{pred_shears}\n{kernel_shears}"

    pred_div = grid_kernel([3]*dim, div=1, **backend)
    kernel_div = torch.as_tensor(kernels.div, **backend).movedim(0, -1).movedim(0, -1)
    assert torch.allclose(pred_div, kernel_div), f"{pred_div}\n{kernel_div}"


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
def test_vel2mom(device, dim):

    device = init_device(device)
    backend = dict(device=device, dtype=torch.float32)

    basis = torch.zeros([5]*dim + [dim, dim], **backend)
    for d in range(dim):
        basis[(2,)*dim + (d, d)] = 1

    kernel_absolute = grid_kernel([5]*dim, absolute=1, **backend)
    pred_absolute = torch.stack(
        [grid_vel2mom(basis[..., d], absolute=1)[..., d] for d in range(dim)],
        -1)
    assert torch.allclose(kernel_absolute, pred_absolute)

    kernel_membrane = grid_kernel([5]*dim, membrane=1, **backend)
    pred_membrane = torch.stack(
        [grid_vel2mom(basis[..., d], membrane=1)[..., d] for d in range(dim)],
        -1)
    assert torch.allclose(kernel_membrane, pred_membrane)

    kernel_bending = grid_kernel([5]*dim, bending=1, **backend)
    pred_bending = torch.stack(
        [grid_vel2mom(basis[..., d], bending=1)[..., d] for d in range(dim)],
        -1)
    assert torch.allclose(kernel_bending, pred_bending)

    kernel_shears = grid_kernel([5]*dim, shears=1, **backend)
    pred_shears = torch.stack(
        [grid_vel2mom(basis[..., d], shears=1) for d in range(dim)],
        -1)
    assert torch.allclose(kernel_shears, pred_shears)

    kernel_div = grid_kernel([5]*dim, div=1, **backend)
    pred_div = torch.stack(
        [grid_vel2mom(basis[..., d], div=1) for d in range(dim)],
        -1)
    assert torch.allclose(kernel_div, pred_div)
