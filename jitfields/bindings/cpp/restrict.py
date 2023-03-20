from ..common.utils import cinfo, ctypename, boundspline_template
from ..common.resize import get_shift_scale
from .utils import cwrap
import cppyy
import numpy as np
from .utils import include

include()
cppyy.include('restrict.hpp')


def restrict(out, inp, factor, anchor, order, bound):
    ndim = len(order)
    nbatch = inp.ndim - ndim

    shift, scale = get_shift_scale(anchor, inp.shape[-ndim:], out.shape[-ndim:], factor)

    out.zero_()
    np_inp = inp.numpy()
    np_out = out.numpy()

    scalar_t = np_inp.dtype
    offset_t = np.int64
    reduce_t = np.float64

    inshape, instride = cinfo(np_inp, dtype=offset_t)
    outshape, outstride = cinfo(np_out, dtype=offset_t)

    shift = reduce_t(shift)
    npscale = np.asarray(scale, dtype=reduce_t)

    # dispatch
    if ndim <= 3:
        template = f'{nbatch}, {ndim}'
        template += ', ' + ctypename(scalar_t)
        template += ', ' + ctypename(offset_t)
        template += ', ' + ctypename(reduce_t) + ', '
        template += boundspline_template(bound, order)
        if all(1 < s <= 2 for s in scale):
            func = cppyy.gbl.jf.restrict.loop2
        else:
            func = cppyy.gbl.jf.restrict.loop
        func = cwrap(func[template])
        func(np_out, np_inp, shift, npscale,
             outshape, inshape, outstride, instride)
    else:
        template = f'{nbatch}, {ndim}'
        template += ', ' + ctypename(scalar_t)
        template += ', ' + ctypename(offset_t)
        template += ', ' + ctypename(reduce_t)
        func = cwrap(cppyy.gbl.jf.restrict.loopnd[template])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(np_out, np_inp, shift, npscale, order, bound,
             outshape, inshape, outstride, instride)

    return out, scale
