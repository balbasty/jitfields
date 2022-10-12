import numpy as np


def cinfo(array, dtype=None, backend=np):
    """Return shape and stride as numpy arrays"""
    shape = backend.asarray(array.shape, dtype=dtype)
    stride = backend.asarray(cstrides(array), dtype=dtype)
    return shape, stride


def cstrides(array):
    """Get C strides (in elements, not bytes) from numpy array"""
    itemsize = np.dtype(array.dtype).itemsize
    strides = [s // itemsize for s in array.strides]
    return strides
