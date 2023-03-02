import cppyy
import math as pymath
import torch
import os
import ctypes
import numpy as np
from .utils import include

# cppyy.set_debug(True)
include()
cppyy.include('distance_l1.hpp')
cppyy.include('distance_euclidean.hpp')


def l1dt_1d_(f, dim=-1, w=1):
    """in-place one-dimensional L1 distance"""
    f = f.movedim(dim, -1)

    npf = f.numpy()
    n = pymath.prod(npf.shape[:-1])
    scalar_t = npf.dtype
    itemsize = scalar_t.itemsize
    offset_t = np.int32 if n <= np.iinfo('int32').max else np.int64

    cscalar_t = ctypes.c_float if scalar_t == np.float32 else ctypes.c_double
    coffset_t = ctypes.c_int32 if offset_t == np.int32 else ctypes.c_int64

    shape = np.asarray(npf.shape, dtype=offset_t)
    stride = np.asarray([s // itemsize for s in npf.strides], dtype=offset_t)

    if f.ndim == 3:
        cppyy.gbl.jf.distance_l1.loop3d(
            npf.ctypes.data_as(ctypes.POINTER(cscalar_t)), cscalar_t(float(w)),
            shape.ctypes.data_as(ctypes.POINTER(coffset_t)),
            stride.ctypes.data_as(ctypes.POINTER(coffset_t)))

    else:

        cppyy.gbl.jf.distance_l1.loop(
            npf.ctypes.data_as(ctypes.POINTER(cscalar_t)),
            cscalar_t(float(w)), f.dim(),
            shape.ctypes.data_as(ctypes.POINTER(coffset_t)),
            stride.ctypes.data_as(ctypes.POINTER(coffset_t)))

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
    f = f.movedim(dim, -1)

    npf = f.numpy()
    n = pymath.prod(npf.shape[:-1])
    scalar_t = npf.dtype
    itemsize = scalar_t.itemsize
    offset_t = np.int32 if n <= np.iinfo('int32').max else np.int64
    offset_size = np.dtype(offset_t).itemsize

    buf = f.new_empty([f.shape[-1] * (2 * itemsize + offset_size)],
                      dtype=torch.uint8)

    cscalar_t = ctypes.c_float if scalar_t == np.float32 else ctypes.c_double
    coffset_t = ctypes.c_int32 if offset_t == np.int32 else ctypes.c_int64

    shape = np.asarray(npf.shape, dtype=offset_t)
    stride = np.asarray([s // itemsize for s in npf.strides], dtype=offset_t)

    if f.ndim == 3:
        cppyy.gbl.jf.distance_e.loop3d(
            npf.ctypes.data_as(ctypes.POINTER(cscalar_t)),
            buf.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            cscalar_t(float(w)),
            shape.ctypes.data_as(ctypes.POINTER(coffset_t)),
            stride.ctypes.data_as(ctypes.POINTER(coffset_t)))

    else:
        cppyy.gbl.jf.distance_e.loop(
            npf.ctypes.data_as(ctypes.POINTER(cscalar_t)),
            buf.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            cscalar_t(float(w)), f.dim(),
            shape.ctypes.data_as(ctypes.POINTER(coffset_t)),
            stride.ctypes.data_as(ctypes.POINTER(coffset_t)))

    return f.movedim(-1, dim)


def edt_1d(f, dim=-1, w=1):
    """out-of-place one-dimensional Euclidean distance"""
    dtype = f.dtype
    if not f.dtype.is_floating_point:
        f = torch.get_default_dtype()
    f = f.to(dtype, copy=True)
    return edt_1d_(f, dim, w)
