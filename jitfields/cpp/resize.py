from ..common.bounds import convert_bound
from ..common.spline import convert_order
from ..utils import ensure_list
from .utils import boundspline_template, cwrap, cinfo
import cppyy
import numpy as np
import os

this_folder = os.path.abspath(os.path.dirname(__file__))
cppyy.add_include_path(os.path.join(this_folder, '..', 'csrc'))
cppyy.include('resize.hpp')


def resize(x, factor=None, shape=None, ndim=None,
           anchor='e', order=2, bound='dct2', out=None):
    order = [convert_order.get(o, o) for o in ensure_list(order, ndim)]
    bound = [convert_bound.get(b, b) for b in ensure_list(bound, ndim)]

    anchor = anchor[0].lower()
    if anchor == 'e':
        shift = 0.5
        scale = [si / so for si, so in zip(x.shape[-ndim:], shape)]
    elif anchor == 'c':
        shift = 0
        scale = [(si - 1) / (so - 1) for si, so in zip(x.shape[-ndim:], shape)]
    else:
        shift = 0
        scale = [1/f for f in factor]

    fullshape = list(x.shape[:-ndim]) + list(shape)
    out = x.new_empty(fullshape) if out is None else out.view(fullshape)
    npx = x.numpy()
    npy = out.numpy()

    offset_t = np.int64
    inshape, instride = cinfo(npx, dtype=offset_t)
    outshape, outstride = cinfo(npy, dtype=offset_t)

    scalar_t = npx.dtype.type
    shift = scalar_t(shift)
    scale = np.asarray(scale, dtype=scalar_t)
    nalldim = int(npy.ndim)

    # dispatch
    if ndim <= 3:
        template = boundspline_template(bound, order)
        func = cwrap(getattr(cppyy.gbl.jf.resize, f'loop{ndim}d')[template])
        func(npy, npx, nalldim, shift, scale,
             outshape, inshape, outstride, instride)
    else:
        func = cwrap(cppyy.gbl.jf.resize.loopnd[f'{int(ndim)}'])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(npy, npx, nalldim, shift, scale,
             order, bound, outshape, inshape, outstride, instride)

    return out
