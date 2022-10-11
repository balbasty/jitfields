import ctypes
import numpy as np
from ..common.bounds import convert_bound, cnames as cnames_bound
from ..common.spline import convert_order, cnames as cnames_spline


def ctype(dtype):
    """Convert numpy dtype to C type"""
    dtype = np.dtype(dtype)
    if dtype.name == 'float32':
        return ctypes.c_float
    elif dtype.name == 'float64':
        return ctypes.c_double
    return getattr(ctypes, 'c_' + dtype.name)


def as_pointer(array, dtype=None):
    """Convert numpy array to C pointer"""
    dtype = ctype(dtype or array.dtype)
    return array.ctypes.data_as(ctypes.POINTER(dtype))


def cstrides(array):
    """Get C strides (in elements, not bytes) from numpy array"""
    itemsize = np.dtype(array.dtype).itemsize
    strides = [s // itemsize for s in array.strides]
    return strides


def bound_as_cname(bound):
    """Get C bound type"""
    if isinstance(bound, (list, tuple)):
        bound = [bound_as_cname(b) for b in bound]
    else:
        bound = convert_bound.get(bound, bound)
        bound = 'jf::bound::type::' + cnames_bound[bound]
    return bound


def spline_as_cname(order):
    """Get C spline type"""
    if isinstance(order, (list, tuple)):
        order = [spline_as_cname(o) for o in order]
    else:
        order = convert_order.get(order, order)
        order = 'jf::spline::type::' + cnames_spline[order]
    return order


def boundspline_template(bound, order):
    """Generate C template from bound/spline"""
    assert len(bound) == len(order)
    if not (isinstance(bound[0], str) and bound[0].startswith('jf::')):
        bound = bound_as_cname(bound)
    if not (isinstance(order[0], str) and order[0].startswith('jf::')):
        order = spline_as_cname(order)
    tpl = ''
    for o, b in zip(order, bound):
        tpl += f'{o}, {b}, '
    tpl = tpl[:-2]
    return tpl


def as_ctype(x):
    """Convert arrays to C pointers and scalars to ctypes scalars"""
    if isinstance(x, np.ndarray):
        return as_pointer(x)
    return ctype(type(x))(x)


def cwrap(func):
    """Decorator to automatically cast inputs to a cppyy function"""
    def call(*args):
        args = list(map(as_ctype, args))
        args = [int(a.value) if isinstance(a, ctypes.c_int) else
                float(a.value) if isinstance(a, ctypes.c_float) else a
                for a in args]
        return func(*args)
    return call


def cinfo(array, dtype=None):
    """Return shape and stride as numpy arrays"""
    shape = np.asarray(array.shape, dtype=dtype)
    stride = np.asarray(cstrides(array), dtype=dtype)
    return shape, stride


def ctypename(dtype):
    dtype = ctype(dtype).__name__[2:]
    if dtype == 'byte':
        return 'signed char'
    if dtype == 'ubyte':
        return 'unsigned char'
    if dtype[0] == 'u':
        return 'unsigned ' + dtype[1:]
    return dtype


