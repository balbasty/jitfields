from ..common.bounds import convert_bound
from ..common.spline import convert_order
from ..common.utils import cinfo, bound_as_cname, ctypename
from .utils import (culaunch, get_offset_type, to_cupy, CachedKernel)
from ..common.splinc import get_poles
import cupy as cp


def get_kernel(key):
    nbatch, npoles, bound, scalar_t, offset_t, reduce_t = key
    template = f'kernel<{nbatch}, {npoles}, {bound_as_cname(bound)}, '
    template += ctypename(scalar_t) + ', '
    template += ctypename(offset_t) + ', '
    template += ctypename(reduce_t) + '>'
    return template


kernels = CachedKernel('splinc.cu', get_kernel)


def spline_coeff_(inp, order, bound, dim=-1):
    nbatch = inp.ndim - 1
    order = convert_order.get(order, order)
    bound = convert_bound.get(bound, bound)
    if order in (0, 1):
        return inp

    cu = to_cupy(inp.movedim(dim, -1))
    offset_t = get_offset_type(cu)
    scalar_t = cu.dtype.type
    reduce_t = cp.double
    shape, stride = cinfo(cu, dtype=offset_t, backend=cp)

    poles = cp.asarray(get_poles(order), dtype=reduce_t)
    npoles = len(poles)

    # dispatch
    numel = inp.movedim(dim, -1).shape[:-1].numel()
    kernel = kernels.get(nbatch, npoles, bound, scalar_t, offset_t, reduce_t)
    culaunch(kernel, numel, (cu, shape, stride, poles))

    return inp


def spline_coeff_nd_(inp, order, bound, ndim=None):
    if ndim is None:
        ndim = inp.dim()

    if all([o in (0, 1) for o in order]):
        return inp

    for d, b, o in zip(range(ndim), reversed(bound), reversed(order)):
        spline_coeff_(inp, o, b, dim=-d-1)

    return inp
