from ..common.utils import cinfo, ctypename, boundspline_template
from ..common.resize import get_shift_scale
from .utils import (get_offset_type, to_cupy, culaunch, CachedKernel)
import cupy as cp


def get_kernel(*key):
    """1/2/3D kernels"""
    nbatch, ndim, order, bound, scalar_t, offset_t, reduce_t = key
    template = f'kernel<{nbatch}, {ndim}, '
    template += ctypename(scalar_t) + ', '
    template += ctypename(offset_t) + ', '
    template += ctypename(reduce_t) + ', '
    template += boundspline_template(bound, order)
    template += '>'
    return template


kernels = CachedKernel('resize.cu', get_kernel)
ndkernels = CachedKernel('resize.cu')


def resize(out, x, factor, anchor, order, bound):
    ndim = len(order)
    nbatch = x.ndim - ndim
    numel = out.shape.numel()

    shift, scale = get_shift_scale(anchor, x.shape[-ndim:], out.shape[:-ndim], factor)

    cux = to_cupy(x)
    cuy = to_cupy(out)

    scalar_t = cux.dtype.dtype
    reduce_t = scalar_t
    offset_t = get_offset_type(cux, cuy)

    inshape, instride = cinfo(cux, dtype=offset_t, backend=cp)
    outshape, outstride = cinfo(cuy, dtype=offset_t, backend=cp)

    shift = reduce_t(shift)
    scale = cp.asarray(scale, dtype=reduce_t)

    # dispatch
    if ndim <= 3:
        kernel = kernels.get(nbatch, ndim, tuple(order), tuple(bound),
                             scalar_t, offset_t, reduce_t)
        args = (cuy, cux, shift, scale, outshape, inshape, outstride, instride)
        culaunch(kernel, numel, args)
    else:
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        kernel = ndkernels.get(nbatch, ndim, scalar_t, offset_t, reduce_t)
        args = (cuy, cux, shift, scale, order, bound, outshape, inshape, outstride, instride)
        culaunch(kernel, numel, args)

    return out
