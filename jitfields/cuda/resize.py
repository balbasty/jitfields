import os
from .bounds import code as bounds_code, convert_bound
from .interpolation import code as interpolation_code
from .batch import code as batch_code
from ..utils import ensure_list, prod
from .utils import get_cuda_blocks, get_cuda_num_threads
import itertools
import math as pymath
from torch.utils.dlpack import to_dlpack
import cupy as cp
import torch


code = ''
code += bounds_code + '\n'
code += interpolation_code + '\n'
code += batch_code + '\n'

this_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_folder, 'resize.cu'), 'rt') as f:
    code += f.read() + '\n'

orders = ['Nearest', 'Linear', 'Quadratic', 'Cubic',
          'FourthOrder', 'FifthOrder', 'SixthOrder', 'SeventhOrder']
bounds = ['Zero', 'Replicate', 'DCT1', 'DCT2',
          'DST1', 'DST2', 'DFT', 'NoCheck']

templates = []
keys = []
for ndim in (1,):
    kernel0 = f'kernel{ndim}d<'
    for order in itertools.product(range(4), repeat=ndim):
        for bound in itertools.product(range(8), repeat=ndim):
            kernel1 = kernel0
            for o, b in zip(order, bound):
                kernel1 = kernel1 + f'interpolation::type::{orders[o]}, bound::type::{bounds[b]}, '
            for scalar_t in ('float', 'double'):
                for offset_t in ('int', 'long'):
                    kernel2 = kernel1 + f'{scalar_t}, {offset_t}>'
                    templates += [kernel2]
                    scalar_np = cp.float32 if scalar_t == 'float' else cp.float64
                    offset_np = cp.int32 if offset_t == 'int' else cp.int64
                    keys += [(ndim, tuple(order), tuple(bound), scalar_np, offset_np)]

module = cp.RawModule(code=code, options=('--std=c++14',),
                      name_expressions=tuple(templates))


class GetKernel:
    def __init__(self, template):
        self.template = template

    def __call__(self):
        return module.get_function(self.template)


kernels = {key: GetKernel(template)
           for key, template in zip(keys, templates)}


def resize(x, factor=None, shape=None, ndim=None,
           anchor='e', order=2, bound='dct2'):
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
        raise ValueError('At leat one of shape or factor must be provided')
    if not shape:
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

    if (prod(x.shape) < cp.iinfo('int32').max
            and prod(shape) < cp.iinfo('int32').max):
        offset_t = cp.int32
    else:
        offset_t = cp.int64

    cux = cp.from_dlpack(to_dlpack(x))
    inshape = cp.asarray(cux.shape, dtype=offset_t)
    instride = [s // cp.dtype(cux.dtype).itemsize for s in cux.strides]
    instride = cp.asarray(instride, dtype=offset_t)

    y = x.new_empty(list(cux.shape[:-ndim]) + list(shape))
    cuy = cp.from_dlpack(to_dlpack(y))
    outshape = cp.asarray(cuy.shape, dtype=offset_t)
    outstride = [s // cp.dtype(cux.dtype).itemsize for s in cuy.strides]
    outstride = cp.asarray(outstride, dtype=offset_t)

    dtype = cp.dtype(cux.dtype)
    shift = dtype.type(shift)
    scale = cp.asarray(scale, dtype=dtype)

    # dispatch
    kernel = kernels[(ndim, tuple(order), tuple(bound), dtype.type, offset_t)]()
    kernel((get_cuda_blocks(prod(cuy.shape)),), (get_cuda_num_threads(),),
           (cuy, cux, cp.int(cux.ndim), shift, scale,
            outshape, inshape, outstride, instride))

    return y