import os
import torch
from torch.utils.dlpack import (
    to_dlpack as torch_to_dlpack, 
    from_dlpack as torch_from_dlpack,
)
import numpy as np
from ...utils import prod
from ..common.utils import ctypename
import re
import cupy as cp
from cupy_backends.cuda.api.driver import CUDADriverError

try:
    from cupy import (
        from_dlpack as cupy_from_dlpack, 
        to_dlpack as cupy_to_dlpack,
    )
except ImportError:
    import cupy
    from cupy import fromDlpack as cupy_from_dlpack
    cupy_to_dlpack = cupy.ndarray.toDlpack


_cuda_num_threads = os.environ.get('CUDA_NUM_THREADS', 1024)
_num_threads = torch.get_num_threads()


def init_cuda_num_threads():
    set_cuda_num_threads(None)


def set_cuda_num_threads(n):
    globals()['_cuda_num_threads'] = n


def get_cuda_num_threads():
    return globals()['_cuda_num_threads'] or os.environ.get('CUDA_NUM_THREADS', 1024)


def init_num_threads():
    set_num_threads(None)


def set_num_threads(n):
    globals()['_num_threads'] = n


def get_num_threads():
    return globals()['_num_threads'] or torch.get_num_threads()


def get_cuda_blocks(n, max_threads_per_block=None):
    max_threads_per_block = max_threads_per_block or get_cuda_num_threads()
    return (n - 1) // max_threads_per_block + 1


def culaunch(kernel, numel, args, num_threads=None, num_blocks=None):
    trials = 0
    num_threads = num_threads or get_cuda_num_threads()
    e = None
    while trials < 4 and num_threads >= 1:
        num_blocks = num_blocks or get_cuda_blocks(numel, num_threads)
        try:
            return kernel((num_blocks,), (num_threads,), args)
        except CUDADriverError as e:
            num_threads = num_threads // 2
    raise e


def get_offset_type(*shapes):
    can_use_32b = True
    for shape in shapes:
        if shape is None:
            continue
        if isinstance(shape, (np.ndarray, cp.ndarray)):
            array = shape
            maxstride = max(sz * st for sz, st in zip(array.shape, array.strides))
        else:
            maxstride = prod(shape)
        if maxstride >= np.iinfo('int32').max:
            can_use_32b = False
            break
    return np.int32 if can_use_32b else np.int64


def load_code(filename, relative=None):
    if not relative:
        relative = os.path.abspath(os.path.dirname(__file__))
        relative = os.path.join(relative, '..', '..', 'csrc', 'cuda')

    filepath = os.path.join(relative, filename)
    dirname = os.path.dirname(filepath)
    with open(filepath) as f:
        code = f.read()
    lines = code.split('\n')

    pattern = re.compile(r'\s*#include\s+"(?P<filename>[^"]+)"')

    code = ''
    for line in lines:
        match = pattern.match(line)
        if match:
            code += load_code(match.group('filename'), relative=dirname)
        else:
            code += line + '\n'
    return code


class LazyLoadCode:
    """Utility to lazily load code. Code is loaded on first call to `str`."""

    def __init__(self, filename):
        self.filename = filename
        self.code = None

    def __str__(self):
        if not self.code:
            self.code = load_code(self.filename)
        return self.code


class CachedKernel:
    """Utility to fetch CUDA kernels"""

    def __init__(self, filename, key2expr=None):
        self.code = LazyLoadCode(filename)
        self.key2expr = key2expr
        self.kernels = {}

    def default_key2expr(self, keys):
        func = keys[0]
        args = []
        for key in keys[1:]:
            if isinstance(key, (np.dtype, torch.dtype, type)):
                args.append(ctypename(key))
            elif isinstance(key, bool):
                args.append('true' if key else 'false')
            else:
                args.append(str(key))
        return f'{func}<{",".join(args)}>'

    def get(self, *key):
        if key not in self.kernels:
            key2expr = self.key2expr or self.default_key2expr
            expr = key2expr(key)
            module = cp.RawModule(
                code=str(self.code),
                options=('--std=c++14', '-default-device'),
                name_expressions=(expr,))
            kernel = module.get_function(expr)
            if int(os.environ.get('JF_CACHE_KERNELS', '1')):
                self.kernels[key] = kernel
        else:
            kernel = self.kernels[key]
        return kernel


def to_cupy(x):
    """Convert a torch tensor to cupy without copy"""
    return cupy_from_dlpack(torch_to_dlpack(x))


def from_cupy(x):
    """Convert a cupy tensor to torch without copy"""
    return torch_from_dlpack(cupy_to_dlpack(x))
