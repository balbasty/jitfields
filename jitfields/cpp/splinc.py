from .utils import cwrap
from ..common.utils import cinfo, bound_as_cname
from ..common.splinc import get_poles
import numpy as np
import cppyy
from .utils import include

include()
cppyy.include('splinc.hpp')


def spline_coeff_(inp, order, bound, dim=-1):
    if order in (0, 1):
        return inp
    bound = bound_as_cname(bound)

    np_inp = inp.movedim(dim, -1).numpy()
    shape, stride = cinfo(np_inp, dtype='int64')
    poles = np.asarray(get_poles(order), dtype='double')

    func = cwrap(cppyy.gbl.jf.splinc.loop[bound])
    func(np_inp, int(np_inp.ndim), shape, stride, poles, int(len(poles)))
    return inp


def spline_coeff_nd_(inp, order, bound, ndim=None):
    ndim = ndim or inp.dim()
    for d, b, o in zip(range(ndim), reversed(bound), reversed(order)):
        spline_coeff_(inp, o, b, dim=-d-1)
    return inp
