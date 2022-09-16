import os
from .bounds import code as bounds_code, cnames as bound_names, convert_bound
from .spline import code as spline_code, cnames as order_names, convert_order
from .batch import code as batch_code
from ..utils import ensure_list, prod
from .utils import get_cuda_blocks, get_cuda_num_threads, get_offset_type
import itertools
import math as pymath
from torch.utils.dlpack import to_dlpack
import cupy as cp

# ===
# Build CUDA code
# ===

code = ''
code += bounds_code + '\n'
code += spline_code + '\n'
code += batch_code + '\n'

this_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_folder, 'restrict.cu'), 'rt') as f:
    code += f.read() + '\n'

# ===
# Load module + dispatch 1D/2D/3D
# ===
kernels = {}


def get_kernel(ndim, scale2, order, bound, scalar_t, offset_t):
    key = (ndim, scale2, order, bound, scalar_t, offset_t)
    if key not in kernels:
        if scale2:
            template = f'kernel2{ndim}d<'
        else:
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


def restrict(x, factor=None, shape=None, ndim=None,
             anchor='e', order=1, bound='dct2', out=None):
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
        scale = factor

    fullshape = list(x.shape[:-ndim]) + list(shape)
    if out is None:
        out = x.new_empty(fullshape)
    else:
        out = out.view(fullshape)
    cux = cp.from_dlpack(to_dlpack(x))
    cuy = cp.from_dlpack(to_dlpack(out))
    offset_t = get_offset_type(cux.shape, cuy.shape)

    inshape = cp.asarray(cux.shape, dtype=offset_t)
    instride = [s // cp.dtype(cux.dtype).itemsize for s in cux.strides]
    instride = cp.asarray(instride, dtype=offset_t)

    outshape = cp.asarray(cuy.shape, dtype=offset_t)
    outstride = [s // cp.dtype(cux.dtype).itemsize for s in cuy.strides]
    outstride = cp.asarray(outstride, dtype=offset_t)

    scalar_t = cux.dtype.type
    shift = scalar_t(shift)
    cuscale = cp.asarray(scale, dtype=scalar_t)

    # dispatch
    if ndim <= 3:
        scale2 = all(1 < s <= 2 for s in scale)
        kernel = get_kernel(ndim, scale2, tuple(order), tuple(bound), scalar_t, offset_t)
        kernel((get_cuda_blocks(prod(cuy.shape)),), (get_cuda_num_threads(),),
               (cuy, cux, cp.int(cux.ndim), shift, cuscale,
                outshape, inshape, outstride, instride))
    else:
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        kernel = get_kernelnd(ndim, scalar_t, offset_t)
        kernel((get_cuda_blocks(prod(cuy.shape)),), (get_cuda_num_threads(),),
               (cuy, cux, cp.int(cux.ndim), shift, cuscale, order, bound,
                outshape, inshape, outstride, instride))

    return out, scale
