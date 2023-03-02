from ..common.bounds import convert_bound
from ..common.spline import convert_order
from ..utils import ensure_list
from ..common.utils import cinfo, ctypename, boundspline_template
from .utils import cwrap
import cppyy
import numpy as np
from .utils import include

include()
cppyy.include('restrict.hpp')


def restrict(inp, factor=None, shape=None, ndim=None,
             anchor='e', order=2, bound='dct2', out=None):

    order = [convert_order.get(o, o) for o in ensure_list(order, ndim)]
    bound = [convert_bound.get(b, b) for b in ensure_list(bound, ndim)]

    anchor = anchor[0].lower()
    if anchor == 'e':
        shift = 0.5
        scale = [si / so for si, so in zip(inp.shape[-ndim:], shape)]
    elif anchor == 'c':
        shift = 0
        scale = [(si - 1) / (so - 1) for si, so in zip(inp.shape[-ndim:], shape)]
    else:
        shift = 0
        scale = [1/f for f in factor]

    fullshape = list(inp.shape[:-ndim]) + list(shape)
    if out is None:
        out = inp.new_empty(fullshape)
    else:
        out = out.view(fullshape)

    np_inp = inp.numpy()
    np_out = out.numpy()

    offset_t = np.int64
    inshape, instride = cinfo(np_inp, dtype=offset_t)
    outshape, outstride = cinfo(np_out, dtype=offset_t)

    scalar_t = np_inp.dtype.type
    shift = scalar_t(shift)
    npscale = np.asarray(scale, dtype=scalar_t)
    nalldim = int(np_out.ndim)

    # dispatch
    if ndim <= 3:
        template = f'{nalldim}'
        template += ', ' + boundspline_template(bound, order)
        template += ', ' + ctypename(scalar_t)
        template += ', ' + ctypename(offset_t)
        if all(1 < s <= 2 for s in scale):
            func = getattr(cppyy.gbl.jf.restrict, f'loop2{ndim}d')
        else:
            func = getattr(cppyy.gbl.jf.restrict, f'loop{ndim}d')
        func = cwrap(func[template], 'restrict')
        func(np_out, np_inp, shift, npscale,
             outshape, inshape, outstride, instride)
    else:
        template = f'{ndim}, {nalldim}'
        template += ', ' + ctypename(scalar_t)
        template += ', ' + ctypename(offset_t)
        func = cwrap(cppyy.gbl.jf.restrict.loopnd[template])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(np_out, np_inp, shift, npscale, order, bound,
             outshape, inshape, outstride, instride)

    return out, scale
