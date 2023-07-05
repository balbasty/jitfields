import cppyy
import torch
import numpy as np
from .utils import include, cwrap
from ..common.utils import ctypename, cinfo, spline_as_cname, bound_as_cname

include()
cppyy.include('distance_l1.hpp')
cppyy.include('distance_euclidean.hpp')
cppyy.include('distance_spline.hpp')


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



def splinedt_table_(time, dist, loc, coeff, timetable, order, bound):
    """in-place distance to 1D spline"""
    ndim = coeff.shape[-1]
    batch = coeff.shape[:-2]
    nbatch = len(batch)
    ntimes = timetable.shape[-1]

    nptime = time.numpy()
    npdist = dist.numpy()
    nploc = loc.numpy()
    npcoeff = coeff.numpy()
    nptimetable = timetable.numpy()

    scalar_t = npcoeff.dtype
    offset_t = np.int64
    shape, stride_coeff = cinfo(npcoeff, dtype=offset_t)
    _, stride_time = cinfo(nptime, dtype=offset_t)
    _, stride_dist = cinfo(npdist, dtype=offset_t)
    _, stride_loc = cinfo(nploc, dtype=offset_t)
    _, stride_timetable = cinfo(nptimetable, dtype=offset_t)
    order = spline_as_cname(order)
    bound = bound_as_cname(bound)

    template = f'{nbatch}, {ndim}, {order}, {bound}, {ctypename(scalar_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_spline.mindist_table[template])
    func(nptime, npdist, nploc, npcoeff, nptimetable, int(ntimes), shape, 
         stride_time, stride_dist, stride_loc, stride_coeff, stride_timetable)

    return time, dist


def splinedt_quad_(time, dist, loc, coeff, order, bound, max_iter, tol, step):
    """in-place distance to 1D spline"""
    ndim = coeff.shape[-1]
    batch = coeff.shape[:-2]
    nbatch = len(batch)

    nptime = time.numpy()
    npdist = dist.numpy()
    nploc = loc.numpy()
    npcoeff = coeff.numpy()

    scalar_t = npcoeff.dtype
    offset_t = np.int64
    shape, stride_coeff = cinfo(npcoeff, dtype=offset_t)
    _, stride_time = cinfo(nptime, dtype=offset_t)
    _, stride_dist = cinfo(npdist, dtype=offset_t)
    _, stride_loc = cinfo(nploc, dtype=offset_t)
    order = spline_as_cname(order)
    bound = bound_as_cname(bound)
    max_iter = int(max_iter)
    tol = float(tol)
    step = float(step)

    template = f'{nbatch}, {ndim}, {order}, {bound}, {ctypename(scalar_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_spline.mindist_brent[template])
    func(nptime, npdist, nploc, npcoeff, shape, 
         stride_time, stride_dist, stride_loc, stride_coeff, max_iter, tol, step)
    return time, dist


def splinedt_gaussnewton_(time, dist, loc, coeff, order, bound, max_iter, tol):
    """in-place distance to 1D spline"""
    ndim = coeff.shape[-1]
    batch = coeff.shape[:-2]
    nbatch = len(batch)

    nptime = time.numpy()
    npdist = dist.numpy()
    nploc = loc.numpy()
    npcoeff = coeff.numpy()

    scalar_t = npcoeff.dtype
    offset_t = np.int64
    shape, stride_coeff = cinfo(npcoeff, dtype=offset_t)
    _, stride_time = cinfo(nptime, dtype=offset_t)
    _, stride_dist = cinfo(npdist, dtype=offset_t)
    _, stride_loc = cinfo(nploc, dtype=offset_t)
    order = spline_as_cname(order)
    bound = bound_as_cname(bound)
    max_iter = int(max_iter)
    tol = float(tol)

    template = f'{nbatch}, {ndim}, {order}, {bound}, {ctypename(scalar_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_spline.mindist_gaussnewton[template])
    func(nptime, npdist, nploc, npcoeff, shape, 
         stride_time, stride_dist, stride_loc, stride_coeff, max_iter, tol)
    return time, dist
