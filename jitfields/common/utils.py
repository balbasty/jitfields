import numpy as np


def cinfo(array, dtype=None, backend=np):
    """Return shape and strides as numpy arrays"""
    shape = backend.asarray(array.shape, dtype=dtype)
    stride = backend.asarray(strides_np2c(array), dtype=dtype)
    return shape, stride


def cstrides(array, dtype=None, backend=np):
    """Return strides as numpy array"""
    return backend.asarray(strides_np2c(array), dtype=dtype)


def cshape(array, dtype=None, backend=np):
    """Return shape as numpy array"""
    return backend.asarray(array.shape, dtype=dtype)


def strides_np2c(array):
    """Get C strides (in elements, not bytes) from numpy array"""
    itemsize = np.dtype(array.dtype).itemsize
    strides = [s // itemsize for s in array.strides]
    return strides
