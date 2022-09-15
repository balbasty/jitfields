import os
from .bounds import code as bounds_code, cnames as bound_names, convert_bound
from .interpolation import code as interpolation_code, cnames as order_names
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
code += interpolation_code + '\n'
code += batch_code + '\n'

this_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_folder, 'resize.cu'), 'rt') as f:
    code += f.read() + '\n'

# ===
# Load module with 1D/2D/3D kernels
# ===

templates = []
keys = []
for ndim in (1, 2, 3):
    kernel0 = f'kernel{ndim}d<'
    for order in itertools.product(range(4), repeat=ndim):
        for bound in itertools.product(range(8), repeat=ndim):
            kernel1 = kernel0
            for o, b in zip(order, bound):
                kernel1 += f'interpolation::type::{order_names[o]}, '
                kernel1 += f'bound::type::{bound_names[b]}, '
            for scalar_t in ('float', 'double'):
                for offset_t in ('int', 'long'):
                    kernel2 = kernel1 + f'{scalar_t}, {offset_t}>'
                    templates += [kernel2]
                    scalar_np = cp.float32 if scalar_t == 'float' else cp.float64
                    offset_np = cp.int32 if offset_t == 'int' else cp.int64
                    keys += [(ndim, tuple(order), tuple(bound), scalar_np, offset_np)]

module = cp.RawModule(code=code, options=('--std=c++14',),
                      name_expressions=tuple(templates))

# ===
# Dispatch 1D/2D/3D
# We store functions that return the correct kernel in a dictionary.
# The dictionary's keys are the template arguments.
# Note that each kernel will be JIT compiled on first use
# (we do not precompile at import time)
# ===


class GetKernel:
    def __init__(self, template):
        self.template = template

    def __call__(self):
        return module.get_function(self.template)


kernels = {key: GetKernel(template)
           for key, template in zip(keys, templates)}

# ===
# Load module with ND kernels + dispatch
# We do not want to have all possible dimensions hard-coded.
# Instead, we load a module with only the required dimension on dispatch.
# To avoid recompiling the same kernel multiple times, we save modules
# that have been already loaded.
# ===

ndkernels = {}


def get_kernelnd(ndim, scalar_t, offset_t):
    """N-dimensional kernel"""
    key = (ndim, scalar_t, offset_t)
    if key not in ndkernels:
        template = f'kernelnd<{ndim},'
        if scalar_t == cp.float32:
            template += 'int,'
        elif scalar_t == cp.float64:
            template += 'long,'
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
    if not ndim:
        if shape and hasattr(shape, '__len__'):
            ndim = len(shape)
        elif factor and hasattr(factor, '__len__'):
            factor = len(shape)
        else:
            ndim = x.dim()
    if shape:
        shape = ensure_list(shape, ndim)
    elif factor:
        factor = ensure_list(factor, ndim)
    else:
        raise ValueError('At least one of shape or factor must be provided')
    if not shape:
        if out is not None:
            shape = out.shape[-ndim:]
        else:
            shape = [pymath.ceil(s*f) for s, f in zip(x.shape[-ndim:], factor)]

    order = ensure_list(order, ndim)
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
    scale = cp.asarray(scale, dtype=scalar_t)

    # dispatch
    if ndim <= 3:
        kernel = kernels[(ndim, tuple(order), tuple(bound), scalar_t, offset_t)]()
        kernel((get_cuda_blocks(prod(cuy.shape)),), (get_cuda_num_threads(),),
               (cuy, cux, cp.int(cux.ndim), shift, scale,
                outshape, inshape, outstride, instride))
    else:
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        kernel = get_kernelnd(ndim, scalar_t, offset_t)
        kernel((get_cuda_blocks(prod(cuy.shape)),), (get_cuda_num_threads(),),
               (cuy, cux, cp.int(cux.ndim), shift, scale, order, bound,
                outshape, inshape, outstride, instride))

    return out
