from ..common.utils import cinfo, ctypename, boundspline_template
from ..common.resize import get_shift_scale
from .utils import (get_offset_type, to_cupy, culaunch, CachedKernel, prod)
import cupy as cp


def get_kernel(key):
    nbatch, ndim, scale2, order, bound, scalar_t, offset_t, reduce_t = key
    template = 'kernel'
    if scale2:
        template += '2'
    template += f'<{nbatch}, {ndim}, '
    template += ctypename(scalar_t) + ', '
    template += ctypename(offset_t) + ', '
    template += ctypename(reduce_t) + ', '
    template += boundspline_template(bound, order)
    template += '>'
    return template


kernels = CachedKernel('restrict.cu', get_kernel)
ndkernels = CachedKernel('restrict.cu')


# ===
# Main function (format arguments and dispatch)
# ===


def restrict(out, x, factor, anchor, order, bound):
    ndim = len(order)
    nbatch = x.ndim - ndim

    padding = [(int(o) + 1)//2 for o in order]
    fullshape = list(out.shape)
    fullshape[-ndim:] = [s+2*p for s, p in zip(fullshape[-ndim:], padding)]
    numel = prod(fullshape)

    shift, scale = get_shift_scale(anchor, x.shape[-ndim:], out.shape[-ndim:], factor)

    cux = to_cupy(x)
    cuy = to_cupy(out)

    scalar_t = cux.dtype
    reduce_t = scalar_t
    offset_t = get_offset_type(cux, cuy)

    inshape, instride = cinfo(cux, dtype=offset_t, backend=cp)
    outshape, outstride = cinfo(cuy, dtype=offset_t, backend=cp)

    asreduce= (cp.float16 if reduce_t == cp.float16 else
               cp.float32 if reduce_t == cp.float32 else
               cp.float64)

    shift = asreduce(shift)
    cuscale = cp.asarray(scale, dtype=reduce_t)

    # dispatch
    if ndim <= 3:
        scale2 = all(1 < s <= 2 for s in scale)
        kernel = kernels.get(nbatch, ndim, scale2, tuple(order), tuple(bound),
                             scalar_t, offset_t, reduce_t)
        args = (cuy, cux, shift, cuscale, outshape, inshape, outstride, instride)
        culaunch(kernel, numel, args)
    else:
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        kernel = ndkernels.get(nbatch, ndim, scalar_t, offset_t, reduce_t)
        args = (cuy, cux, shift, cuscale, order, bound, outshape, inshape, outstride, instride)
        culaunch(kernel, numel, args)

    return out, scale
