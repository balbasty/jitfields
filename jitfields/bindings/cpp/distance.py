import cppyy
import torch
import numpy as np
from .utils import include, cwrap
from ..common.utils import ctypename, cinfo

include()
cppyy.include('distance_l1.hpp')
cppyy.include('distance_euclidean.hpp')


def l1dt_1d_(f, dim=-1, w=1):
    """in-place one-dimensional L1 distance"""
    ndim = f.ndim
    npf = f.movedim(dim, -1).numpy()

    scalar_t = npf.dtype
    offset_t = np.int64
    shape, stride = cinfo(npf, dtype=offset_t)

    template = f'{ndim}, {ctypename(scalar_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_l1.loop[template])
    func(npf, float(w), shape, stride)

    return f


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
    npf = f.movedim(dim, -1).numpy()

    scalar_t = npf.dtype
    offset_t = np.int64
    shape, stride = cinfo(npf, dtype=offset_t)

    template = f'{ndim}, {ctypename(scalar_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_e.loop[template])
    func(npf, float(w), shape, stride)

    return f


def edt_1d(f, dim=-1, w=1):
    """out-of-place one-dimensional Euclidean distance"""
    dtype = f.dtype
    if not f.dtype.is_floating_point:
        f = torch.get_default_dtype()
    f = f.to(dtype, copy=True)
    return edt_1d_(f, dim, w)
