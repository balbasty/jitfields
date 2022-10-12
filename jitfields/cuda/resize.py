from ..common.bounds import cnames as bound_names, convert_bound
from ..common.spline import cnames as order_names, convert_order
from ..common.utils import cinfo
from ..utils import ensure_list, prod
from .utils import (get_offset_type, load_code, to_cupy, culaunch)
import cupy as cp

# ===
# Load module + dispatch 1D/2D/3D
# ===
code = load_code('resize.cu')
kernels = {}


def get_kernel(ndim, order, bound, scalar_t, offset_t):
    key = (ndim, order, bound, scalar_t, offset_t)
    if key not in kernels:
        template = f'kernel{ndim}d<'
        for o, b in zip(order, bound):
            template += f'spline::type::{order_names[o]}, '
            template += f'bound::type::{bound_names[b]}, '
        if scalar_t == cp.float32:
            template += 'float,'
        elif scalar_t == cp.float64:
            template += 'double,'
        elif scalar_t == cp.float16:
            template += 'half,'
        else:
            raise ValueError('Unknown scalar type', scalar_t)
        if offset_t == cp.int32:
            template += 'int>'
        elif offset_t == cp.int64:
            template += 'long>'
        else:
            raise ValueError('Unknown offset type', offset_t)
        module = cp.RawModule(code=code, options=('--std=c++14',),
                              name_expressions=(template,))
        kernel = module.get_function(template)
        kernels[key] = kernel
    else:
        kernel = kernels[key]

    return kernel


# ===
# Load module + dispatch ND
# ===

ndkernels = {}


def get_kernelnd(ndim, scalar_t, offset_t):
    """N-dimensional kernel"""
    key = (ndim, scalar_t, offset_t)
    if key not in ndkernels:
        template = f'kernelnd<{ndim},'
        if scalar_t == cp.float32:
            template += 'float,'
        elif scalar_t == cp.float64:
            template += 'double,'
        elif scalar_t == cp.float16:
            template += 'half,'
        else:
            raise ValueError('Unknown scalar type', scalar_t)
        if offset_t == cp.int32:
            template += 'int>'
        elif offset_t == cp.int64:
            template += 'long>'
        else:
            raise ValueError('Unknown offset type', offset_t)
        module = cp.RawModule(code=code, options=('--std=c++14',),
                              name_expressions=(template,))
        kernel = module.get_function(template)
        ndkernels[key] = kernel
    else:
        kernel = ndkernels[key]

    return kernel


# ===
# Main function (format arguments and dispatch)
# ===


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

    out = x.new_empty(fullshape) if out is None else out.view(fullshape)
    cux = to_cupy(x)
    cuy = to_cupy(out)
    offset_t = get_offset_type(cux.shape, cuy.shape)

    inshape, instride = cinfo(cux, dtype=offset_t, backend=cp)
    outshape, outstride = cinfo(cuy, dtype=offset_t, backend=cp)

    scalar_t = cux.dtype.type
    shift = scalar_t(shift)
    scale = cp.asarray(scale, dtype=scalar_t)

    # dispatch
    if ndim <= 3:
        kernel = get_kernel(ndim, tuple(order), tuple(bound), scalar_t, offset_t)
        culaunch(kernel, prod(cuy.shape),
                 (cuy, cux, cp.int(cux.ndim), shift, scale,
                  outshape, inshape, outstride, instride))
    else:
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        kernel = get_kernelnd(ndim, scalar_t, offset_t)
        culaunch(kernel, prod(cuy.shape),
                 (cuy, cux, cp.int(cux.ndim), shift, scale, order, bound,
                  outshape, inshape, outstride, instride))

    return out
