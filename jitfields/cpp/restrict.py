from ..common.bounds import convert_bound, cnames as cnames_bound
from ..common.spline import convert_order, cnames as cnames_spline
from ..utils import ensure_list
import cppyy
import numpy as np
import ctypes
import os

this_folder = os.path.abspath(os.path.dirname(__file__))
cppyy.add_include_path(os.path.join(this_folder, '..', 'csrc'))
cppyy.include('restrict.hpp')


def restrict(x, factor=None, shape=None, ndim=None,
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
    if out is None:
        out = x.new_empty(fullshape)
    else:
        out = out.view(fullshape)
    npx = x.numpy()
    npy = out.numpy()
    offset_t = np.int64

    inshape = np.asarray(npx.shape, dtype=offset_t)
    instride = [s // np.dtype(npx.dtype).itemsize for s in npx.strides]
    instride = np.asarray(instride, dtype=offset_t)

    outshape = np.asarray(npy.shape, dtype=offset_t)
    outstride = [s // np.dtype(npy.dtype).itemsize for s in npy.strides]
    outstride = np.asarray(outstride, dtype=offset_t)

    scalar_t = npx.dtype.type
    shift = scalar_t(shift)
    npscale = np.asarray(scale, dtype=scalar_t)

    cscalar_t = ctypes.c_float if scalar_t == np.float32 else ctypes.c_double
    coffset_t = ctypes.c_int32 if offset_t == np.int32 else ctypes.c_int64

    nalldim = npy.ndim

    # dispatch
    if ndim <= 3:
        scale2 = all(1 < s <= 2 for s in scale)
        order = ['jf::spline::type::' + cnames_spline[o] for o in order]
        bound = ['jf::bound::type::' + cnames_bound[b] for b in bound]
        if scale2:
            func = getattr(cppyy.gbl.jf.restrict, f'loop2{ndim}d')
        else:
            func = getattr(cppyy.gbl.jf.restrict, f'loop{ndim}d')
        if ndim <= 1:
            func = func[f'{order[0]}, {bound[0]}']
        elif ndim <= 2:
            func = func[f'{order[0]}, {bound[0]}, '
                        f'{order[1]}, {bound[1]}']
        else:
            func = func[f'{order[0]}, {bound[0]}, '
                        f'{order[1]}, {bound[1]}, '
                        f'{order[2]}, {bound[2]}']
        func(npy.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             npx.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             int(nalldim), cscalar_t(shift),
             npscale.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             outshape.ctypes.data_as(ctypes.POINTER(coffset_t)),
             inshape.ctypes.data_as(ctypes.POINTER(coffset_t)),
             outstride.ctypes.data_as(ctypes.POINTER(coffset_t)),
             instride.ctypes.data_as(ctypes.POINTER(coffset_t)))
    else:
        func = cppyy.gbl.jf.restrict.loopnd[f'{int(ndim)}']
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(npy.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             npx.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             int(nalldim), cscalar_t(shift),
             npscale.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             order.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             bound.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             outshape.ctypes.data_as(ctypes.POINTER(coffset_t)),
             inshape.ctypes.data_as(ctypes.POINTER(coffset_t)),
             outstride.ctypes.data_as(ctypes.POINTER(coffset_t)),
             instride.ctypes.data_as(ctypes.POINTER(coffset_t)))

    return out, scale
