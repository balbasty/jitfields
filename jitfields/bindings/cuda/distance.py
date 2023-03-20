from .utils import (
    culaunch, to_cupy, get_cuda_num_threads, get_cuda_blocks,
    get_offset_type, CachedKernel)
from ..common.utils import cinfo
import cupy as cp
import torch


kernels_l1 = CachedKernel('distance_l1.cu')
kernels_l2 = CachedKernel('distance_euclidean.cu')


def l1dt_1d_(f, dim=-1, w=1):
    """in-place one-dimensional L1 distance"""
    f = f.movedim(dim, -1)
    ndim = f.ndim
    n = f.shape[:-1].numel()

    cuf = to_cupy(f)

    scalar_t = cuf.dtype
    offset_t = get_offset_type(cuf)
    shape, stride = cinfo(cuf, dtype=offset_t, backend=cp)

    asscalar = (cp.float16 if scalar_t == cp.float16 else
                cp.float32 if scalar_t == cp.float32 else
                cp.float64)

    kernel = kernels_l1.get('kernel', ndim, scalar_t, offset_t)
    culaunch(kernel, n, (cuf, asscalar(w), shape, stride))
    return f.movedim(-1, dim)


def l1dt_1d(f, dim=-1, w=1):
    """out-of-place one-dimensional L1 distance"""
    dtype = f.dtype
    if not f.dtype.is_floating_point:
        f = torch.get_default_dtype()
    f = f.to(dtype, copy=True)
    return l1dt_1d_(f, dim, w)


def edt_1d_(f, dim=-1, w=1):
    """in-place one-dimensional Euclidean distance"""
    ndim = f.ndim
    f = f.movedim(dim, -1)
    n = f.shape[:-1].numel()

    cuf = to_cupy(f)

    scalar_t = cuf.dtype
    offset_t = get_offset_type(cuf)
    shape, stride = cinfo(cuf, dtype=offset_t, backend=cp)

    nb_blocks, nb_threads = get_cuda_blocks(n), get_cuda_num_threads()
    buf = nb_blocks * nb_threads * f.shape[-1]
    buf *= 2 * cuf.dtype.itemsize + stride.dtype.itemsize
    buf = cp.empty([buf], dtype=cp.uint8)

    asscalar = (cp.float16 if scalar_t == cp.float16 else
                cp.float32 if scalar_t == cp.float32 else
                cp.float64)

    kernel = kernels_l2.get('kernel', ndim, scalar_t, offset_t)
    culaunch(kernel, n, (cuf, buf, asscalar(w), shape, stride))
    return f.movedim(-1, dim)


def edt_1d(f, dim=-1, w=1):
    """out-of-place one-dimensional Euclidean distance"""
    dtype = f.dtype
    if not f.dtype.is_floating_point:
        f = torch.get_default_dtype()
    f = f.to(dtype, copy=True)
    return edt_1d_(f, dim, w)
