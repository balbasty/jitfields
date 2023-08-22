from jitfields.regularisers import field_kernel, field_matvec
from .ref_kernels import kernels1, kernels2, kernels3
from .utils import get_test_devices, init_device
import torch
import pytest


devices = get_test_devices()
dims = [1, 2, 3]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
def test_kernel(device, dim):

    device = init_device(device)
    backend = dict(device=device, dtype=torch.float32)

    kernel_absolute = torch.zeros([3]*dim, **backend)
    kernel_absolute[(1,)*dim] = 1

    if dim == 1:
        kernels = kernels1
    elif dim == 2:
        kernels = kernels2
    elif dim == 3:
        kernels = kernels3
    else:
        assert False, f"Test not implemented for dim {dim}"

    pred_absolute = field_kernel([3]*dim, absolute=1, **backend)[..., 0]
    kernel_absolute = torch.as_tensor(kernel_absolute, **backend)
    assert torch.allclose(pred_absolute, kernel_absolute), \
           f"{pred_absolute}\n{kernel_absolute}"

    pred_membrane = field_kernel([3]*dim, membrane=1, **backend)[..., 0]
    kernel_membrane = torch.as_tensor(kernels.membrane, **backend)
    assert torch.allclose(pred_membrane, kernel_membrane), \
           f"{pred_membrane}\n{kernel_membrane}"

    pred_bending = field_kernel([5]*dim, bending=1, **backend)[..., 0]
    kernel_bending = torch.as_tensor(kernels.bending, **backend)
    assert torch.allclose(pred_bending, kernel_bending), \
           f"{pred_bending}\n{kernel_bending}"


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
def test_matvec(device, dim):

    device = init_device(device)
    backend = dict(device=device, dtype=torch.float32)

    basis = torch.zeros([5]*dim + [dim, dim], **backend)
    for d in range(dim):
        basis[(2,)*dim + (d, d)] = 1

    kernel_absolute = field_kernel([5]*dim, absolute=1, **backend)
    pred_absolute = torch.stack(
        [field_matvec(dim, basis[..., d], absolute=1)[..., d]
         for d in range(dim)],
        -1)
    assert torch.allclose(kernel_absolute, pred_absolute)

    kernel_membrane = field_kernel([5]*dim, membrane=1, **backend)
    pred_membrane = torch.stack(
        [field_matvec(dim, basis[..., d], membrane=1)[..., d]
         for d in range(dim)],
        -1)
    assert torch.allclose(kernel_membrane, pred_membrane)

    kernel_bending = field_kernel([5]*dim, bending=1, **backend)
    pred_bending = torch.stack(
        [field_matvec(dim, basis[..., d], bending=1)[..., d]
         for d in range(dim)],
        -1)
    assert torch.allclose(kernel_bending, pred_bending)
