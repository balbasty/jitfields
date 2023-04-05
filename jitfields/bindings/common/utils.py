import numpy as np
import torch
from .bounds import convert_bound, cnames as cnames_bound
from .spline import convert_order, cnames as cnames_spline


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


_torch_to_np_dtype = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex32: np.complex64,
    torch.complex64: np.complex128,
    torch.complex128: np.complex256,
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
}


def to_np_dtype(dtype):
    """Convert dtype to np.dtype instance"""
    if isinstance(dtype, torch.dtype):
        dtype = _torch_to_np_dtype[dtype]
    return np.dtype(dtype)


def np_upcast(*dtypes):
    """Compute upcasted type from a bunch of types"""
    if len(dtypes) == 1:
        return dtypes[0]
    elif len(dtypes) == 2:
        left, right = dtypes
        return (np.ones([], dtype=left) * np.ones([], dtype=right)).dtype
    else:
        left, right, *dtypes = dtypes
        return np.upcast(np.upcast(left, right), *dtypes)


def ctypename(dtype):
    dtype = to_np_dtype(dtype)
    if dtype == np.float16:
        return 'half'
    if dtype == np.float32:
        return 'float'
    if dtype == np.float64:
        return 'double'
    if dtype == np.int8:
        return 'signed char'
    if dtype == np.uint8:
        return 'unsigned char'
    if dtype == np.int16:
        return 'short'
    if dtype == np.uint16:
        return 'unsigned short'
    if dtype == np.int32:
        return 'int'
    if dtype == np.uint16:
        return 'unsigned int'
    if dtype == np.int64:
        return 'long'
    if dtype == np.uint64:
        return 'unsigned long'
    if dtype == np.int128:
        return 'long long'
    if dtype == np.uint128:
        return 'unsigned long long'
    raise ValueError('Unsupported datatype', dtype)


def spline_as_cname(order):
    """Get C spline type"""
    if isinstance(order, (list, tuple)):
        order = [spline_as_cname(o) for o in order]
    else:
        order = convert_order.get(order, order)
        order = 'jf::spline::type::' + cnames_spline[order]
    return order


def bound_as_cname(bound):
    """Get C bound type"""
    if isinstance(bound, (list, tuple)):
        bound = [bound_as_cname(b) for b in bound]
    else:
        if isinstance(bound, str):
            bound = bound.lower()
        bound = convert_bound.get(bound, bound)
        bound = 'jf::bound::type::' + cnames_bound[bound]
    return bound


def bound_template(bound):
    """Generate C template from bound"""
    if not (isinstance(bound[0], str) and bound[0].startswith('jf::')):
        bound = bound_as_cname(bound)
    return ', '.join(bound)


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

