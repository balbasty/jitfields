import os
import torch
import numpy as np
from ..utils import prod
import re
from torch.utils.dlpack import to_dlpack
from cupy_backends.cuda.api.driver import CUDADriverError

try:
    from cupy import from_dlpack as cupy_from_dlpack, to_dlpack as cupy_to_dlpack
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


def culaunch(kernel, numel, args):
    trials = 0
    num_threads = get_cuda_num_threads() // 2
    e = None
    while trials < 4 and num_threads >= 1:
        num_blocks = get_cuda_blocks(numel, num_threads)
        try:
            return kernel((num_blocks,), (num_threads,), args)
        except CUDADriverError as e:
            num_threads = num_threads // 2
    raise e


def get_offset_type(*shapes):
    can_use_32b = True
    for shape in shapes:
        if prod(shape) >= np.iinfo('int32').max:
            can_use_32b = False
            break
    return np.int32 if can_use_32b else np.int64


def load_code(filename):
    this_folder = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(this_folder, '..', 'csrc', filename)) as f:
        code = f.read()
    lines = code.split('\n')

    pattern = re.compile(r'\s*#include\s+"(?P<filename>[\w\.]+)"')

    code = ''
    for line in lines:
        match = pattern.match(line)
        if match:
            code += load_code(match.group('filename'))
        else:
            code += line + '\n'
    return code


def to_cupy(x):
    """Convert a torch tensor to cupy without copy"""
    return cupy_from_dlpack(to_dlpack(x))
