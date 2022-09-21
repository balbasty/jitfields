from ..cuda.bounds import cnames as bound_names, convert_bound
from ..cuda.spline import cnames as order_names, convert_order
from ..utils import ensure_list, prod
import cppyy
import numpy as np
import ctypes
import os

cppyy.set_debug(True)
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
    scale = np.asarray(scale, dtype=scalar_t)

    cscalar_t = ctypes.c_float if scalar_t == np.float32 else ctypes.c_double
    coffset_t = ctypes.c_int32 if offset_t == np.int32 else ctypes.c_int64

    nalldim = npy.ndim

    # dispatch
    if ndim <= 3:
        order = [ctypes.c_uint8(int(o)) for o in order]
        bound = [ctypes.c_uint8(int(b)) for b in bound]
        if ndim <= 1:
            func = cppyy.gbl.jf.resize.loop1d[order[0], bound[0]]
        elif ndim <= 2:
            func = cppyy.gbl.jf.resize.loop1d[order[0], bound[0],
                                              order[1], bound[1]]
        else:
            func = cppyy.gbl.jf.resize.loop1d[order[0], bound[0],
                                              order[1], bound[1],
                                              order[2], bound[2]]
        func(npy.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             npx.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             ctypes.c_int32(int(nalldim)), cscalar_t(shift),
             scale.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             outshape.ctypes.data_as(ctypes.POINTER(coffset_t)),
             inshape.ctypes.data_as(ctypes.POINTER(coffset_t)),
             outstride.ctypes.data_as(ctypes.POINTER(coffset_t)),
             instride.ctypes.data_as(ctypes.POINTER(coffset_t)))
    else:
        func = cppyy.gbl.jf.resize.loopnd[ctypes.c_int32(int(ndim))]
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        func(npy.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             npx.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             ctypes.c_int32(int(nalldim)), cscalar_t(shift),
             scale.ctypes.data_as(ctypes.POINTER(cscalar_t)),
             order.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             bound.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             outshape.ctypes.data_as(ctypes.POINTER(coffset_t)),
             inshape.ctypes.data_as(ctypes.POINTER(coffset_t)),
             outstride.ctypes.data_as(ctypes.POINTER(coffset_t)),
             instride.ctypes.data_as(ctypes.POINTER(coffset_t)))

    return out
