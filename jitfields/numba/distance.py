import numba as nb
import numpy as np
import math as pymath
import torch
from . import utils as nbutils


@nb.njit
def l1dt_algo(vec, w):
    tmp = vec[0]
    for q in range(1, len(vec)):
        tmp = min(tmp + w, vec[q])
        vec[q] = tmp
    for q in range(len(vec)-2, -1, -1):
        tmp = min(tmp + w, vec[q])
        vec[q] = tmp


@nb.njit(parallel=True)
def l1dt_kernel_parallel(ptr, shape, stride, dtype, itemsize, w):
    nvox = nbutils.prod(shape[:-1])
    for index in nb.prange(nvox):
        index = np.int64(index)
        batch_offset = nbutils.index2offset(index, shape[:-1], stride[:-1])
        ptr1 = ptr + batch_offset * itemsize
        vec = nbutils.svector(ptr1, shape[-1], stride[-1], dtype)
        l1dt_algo(vec, w)


@nb.njit
def l1dt_kernel(ptr, shape, stride, dtype, itemsize, w):
    if len(shape) > 1:
        return l1dt_kernel_parallel(ptr, shape, stride, dtype, itemsize, w)
    vec = nbutils.svector(ptr, shape[-1], stride[-1], np.dtype(dtype))
    l1dt_algo(vec, w)


def l1dt_1d_(f, dim=-1, w=1):
    """in-place one-dimensional L1 distance"""
    f = f.movedim(dim, -1)
    dtype = np.asarray(f).dtype
    npf = np.asarray(f)
    stride = tuple(s // dtype.itemsize for s in npf.strides)
    l1dt_kernel(npf.ctypes.data, npf.shape, stride,
                dtype, dtype.itemsize, dtype.type(w))
    f = f.movedim(-1, dim)
    return f


def l1dt_1d(f, dim=-1, w=1):
    """out-of-place one-dimensional L1 distance"""
    dtype = f.dtype
    if not f.dtype.is_floating_point:
        f = torch.get_default_dtype()
    f = f.to(dtype, copy=True)
    return l1dt_1d_(f, dim, w)


@nb.njit
def edt_intersection(f, v, w2, k, q):
    vk = v[k]
    fvk = f[vk]
    fq = f[q]
    a, b = q - vk, q + vk
    s = fq - fvk
    s += w2 * (a * b)
    s /= 2 * w2 * a
    return s


@nb.njit
def edt_fillin(f, v, z, d, w2):
    n = len(f)
    k = 0
    for q in range(n):
        while (k < n - 1) and (z[k + 1] < q):
            k += 1
        vk = v[k]
        d[q] = f[vk] + w2 * nbutils.square(q - vk)

    for q in range(n):
        f[q] = d[q]


@nb.njit
def edt_algo(f, v, z, d, w2):
    n = len(f)
    if n == 1:
        return

    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    k = 0
    for q in range(1, n):
        while True:
            s = edt_intersection(f, v, w2, k, q)
            if (k == 0) or (s > z[k]):
                break
            k -= 1

        if pymath.isnan(s):
            s = -np.inf

        k += 1
        v[k] = q
        z[k] = s
        z[k+1] = np.inf

    edt_fillin(f, v, z, d, w2)


@nb.njit(parallel=True)
def edt_kernel_parallel(ptr, shape, stride, dtype, itemsize, w):
    nvox, n = nbutils.prod(shape[:-1]), shape[-1]
    nthreads = nb.get_num_threads()
    z = np.empty(shape=(nthreads, n), dtype=dtype)
    d = np.empty(shape=(nthreads, n), dtype=dtype)
    v = np.empty(shape=(nthreads, n), dtype=np.int64)
    for index in nb.prange(nvox):
        thread_id = nbutils.get_thread_id()
        index = np.int64(index)
        batch_offset = nbutils.index2offset(index, shape[:-1], stride[:-1])
        f = nbutils.svector(ptr + batch_offset * itemsize,
                            shape[-1], stride[-1], dtype)
        edt_algo(f, v[thread_id], z[thread_id], d[thread_id], w)


@nb.njit
def edt_kernel(ptr, shape, stride, dtype, itemsize, w):
    w2 = nbutils.square(w)
    if len(shape) > 1:
        return edt_kernel_parallel(ptr, shape, stride, dtype, itemsize, w2)
    vec = nbutils.svector(ptr, shape[-1], stride[-1], dtype)
    z = np.empty_like(vec)
    d = np.empty_like(vec)
    v = np.empty_like(vec, dtype=np.int64)
    edt_algo(vec, v, z, d, w2)


def edt_1d_(f, dim=-1, w=1):
    """in-place one-dimensional Eudlidean distance"""
    f = f.movedim(dim, -1)
    dtype = np.asarray(f).dtype
    npf = np.asarray(f)
    stride = tuple(s // dtype.itemsize for s in npf.strides)
    itemsize = dtype.itemsize
    w = dtype.type(w)
    edt_kernel(npf.ctypes.data, npf.shape, stride, dtype, itemsize, w)
    f = f.movedim(-1, dim)
    return f


def edt_1d(f, dim=-1, w=1):
    """out-of-place one-dimensional Eudlidean distance"""
    dtype = f.dtype
    if not f.dtype.is_floating_point:
        f = torch.get_default_dtype()
    f = f.to(dtype, copy=True)
    return edt_1d_(f, dim, w)
