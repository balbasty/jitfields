from jitfields.common.bounds import convert_bound, cnames as bound_names
from jitfields.common.spline import convert_order
from ..utils import ensure_list
from ..common.splinc import get_poles
import numpy as np
import ctypes
import cppyy
import os


this_folder = os.path.abspath(os.path.dirname(__file__))
cppyy.add_include_path(os.path.join(this_folder, '..', 'csrc'))
cppyy.include('splinc.hpp')


def spline_coeff_(inp, order, bound, dim=-1):
    order = convert_order.get(order, order)
    bound = 'jf::bound::type::' + bound_names[convert_bound.get(bound, bound)]
    if order in (0, 1):
        return inp

    npx = inp.movedim(dim, -1).numpy()
    offset_t = np.int64
    scalar_t = npx.dtype.type

    shape = np.asarray(npx.shape, dtype=offset_t)
    stride = [s // np.dtype(npx.dtype).itemsize for s in npx.strides]
    stride = np.asarray(stride, dtype=offset_t)
    poles = np.asarray(get_poles(order), dtype='double')

    cscalar_t = ctypes.c_float if scalar_t == np.float32 else ctypes.c_double
    coffset_t = ctypes.c_int64

    func = cppyy.gbl.jf.splinc.loop[bound]
    func(npx.ctypes.data_as(ctypes.POINTER(cscalar_t)), int(npx.ndim),
         shape.ctypes.data_as(ctypes.POINTER(coffset_t)),
         stride.ctypes.data_as(ctypes.POINTER(coffset_t)),
         poles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), int(len(poles)))
    return inp


def spline_coeff_nd_(inp, order, bound, ndim=None):
    if ndim is None:
        ndim = inp.dim()

    order = [convert_order.get(o, o) for o in ensure_list(order, ndim)]
    bound = [convert_bound.get(b, b) for b in ensure_list(bound, ndim)]
    if all([o in (0, 1) for o in order]):
        return inp

    for d, b, o in zip(range(ndim), reversed(bound), reversed(order)):
        spline_coeff_(inp, o, b, dim=-d-1)

    return inp
