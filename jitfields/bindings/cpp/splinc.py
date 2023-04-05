from .utils import cwrap
from ..common.utils import cinfo, bound_as_cname, ctypename
from ..common.splinc import get_poles
import numpy as np
import cppyy
from .utils import include

include()
cppyy.include('splinc.hpp')


def spline_coeff_(inp, order, bound, dim=-1):
    if order in (0, 1):
        return inp
    nbatch = inp.ndim - 1

    np_inp = inp.movedim(dim, -1).numpy()

    scalar_t = np_inp.dtype
    offset_t = np.int64
    reduce_t = np.float64
    shape, stride = cinfo(np_inp, dtype='int64')
    poles = np.asarray(get_poles(order), dtype='double')

    bound = bound_as_cname(bound)
    template = f'{nbatch}, {len(poles)}, {bound}, '
    template += ctypename(scalar_t) + ', '
    template += ctypename(offset_t) + ', '
    template += ctypename(reduce_t)

    func = cwrap(cppyy.gbl.jf.splinc.loop[template])
    func(np_inp, shape, stride, poles)
    return inp


def spline_coeff_nd_(inp, order, bound, ndim=None):
    ndim = ndim or inp.dim()
    for d, b, o in zip(range(ndim), reversed(bound), reversed(order)):
        spline_coeff_(inp, o, b, dim=-d-1)
    return inp
