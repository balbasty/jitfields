import ctypes
import numbers
import cppyy
import os
import numpy as np


def include():
    """Setup include directory"""
    this_folder = os.path.abspath(os.path.dirname(__file__))
    cppyy.add_include_path(os.path.join(this_folder, '..', '..', 'csrc', 'cpp'))


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


def as_ctype(x):
    """Convert arrays to C pointers and scalars to ctypes scalars"""
    if isinstance(x, np.ndarray):
        return as_pointer(x)
    if isinstance(x, (np.number, numbers.Number)):
        return ctype(type(x))(x)
    return x


def nullptr(dtype):
    return ctypes.POINTER(ctype(dtype))()


def cwrap(func):
    """Decorator to automatically cast inputs to a cppyy function"""
    def call(*args):
        args = list(map(lambda x: x if isinstance(x, (int, float)) else as_ctype(x), args))
        out = func(*args)
        return out
    return call
