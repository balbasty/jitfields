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


# def cwrap(func):
#     """Decorator to automatically cast inputs to a cppyy function"""
#     # from time import time
#     def call(*args):
#         args = list(map(lambda x: x if isinstance(x, (int, float)) else as_ctype(x), args))
#         # tic = time()
#         out = func(*args)
#         # print('internal:', (time() - tic) * 1e3, 'ms')
#         return out
#     return call


def cwrap(func, tag='call'):
    """Decorator to automatically cast inputs to a cppyy function"""
    def prep(*args):
        return list(map(lambda x: x if isinstance(x, (int, float)) else as_ctype(x), args))

    def call(*args):
        args = prep(*args)
        out = func(*args)
        return out
    def restrict(*args):
        args = prep(*args)
        out = func(*args)
        return out
    def resize(*args):
        args = prep(*args)
        out = func(*args)
        return out
    def matvec(*args):
        args = prep(*args)
        out = func(*args)
        return out
    def solve(*args):
        args = prep(*args)
        out = func(*args)
        return out
    def solve_(*args):
        args = prep(*args)
        out = func(*args)
        return out
    def vel2mom(*args):
        args = prep(*args)
        out = func(*args)
        return out
    return locals()[tag]
