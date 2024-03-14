from ..common.resize import get_shift_scale
from ..common.utils import cinfo, ctypename, boundspline_template
from .utils import cwrap, include
import cppyy
import numpy as np

include()
cppyy.include('resize.hpp')


def resize(out, x, factor, anchor, order, bound):
    ndim = len(order)
    nbatch = x.ndim - ndim

    shift, scale = get_shift_scale(anchor, x.shape[-ndim:], out.shape[-ndim:], factor)

    npx = x.numpy()
    npy = out.numpy()

    scalar_t = npx.dtype
    offset_t = np.int64
    reduce_t = np.float64

    inshape, instride = cinfo(npx, dtype=offset_t)
    outshape, outstride = cinfo(npy, dtype=offset_t)
    scale = np.asarray(scale, dtype=reduce_t)

    # dispatch
    if ndim <= 3:
        template = f'{nbatch}, {ndim}'
        template += ', ' + ctypename(scalar_t)
        template += ', ' + ctypename(offset_t)
        template += ', ' + ctypename(reduce_t) + ', '
        template += boundspline_template(bound, order)
        func = cwrap(cppyy.gbl.jf.resize.loop[template])
        func(npy, npx, shift, scale,
             outshape, inshape, outstride, instride)
    else:
        template = f'{nbatch}, {ndim}'
        template += ', ' + ctypename(scalar_t)
        template += ', ' + ctypename(offset_t)
        template += ', ' + ctypename(reduce_t)
        func = cwrap(cppyy.gbl.jf.resize.loopnd[template])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(npy, npx, shift, scale,
             order, bound, outshape, inshape, outstride, instride)

    return out
