import os
import torch
from ..import utils
from numba import cuda
from numba.typed import List as nulist
from typing import List, TypeVar, Tuple
T = TypeVar('T')

_cuda_num_threads = None
_num_threads = None


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


remainder = cuda.jit(device=True)(utils.remainder)


@cuda.jit(device=True)
def prod(sequence):
    """Perform the cumulative product of a flat array"""
    accumulate = sequence[0]
    sequence = sequence[1:]
    for elem in sequence:
        accumulate = accumulate * elem
    return accumulate


@cuda.jit(device=True)
def sub2ind(sub: List[int], shape: List[int]) -> int:
    """Convert sub indices (i, j, k) into linear indices.

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    sub : list[int]
    shape : list[int]

    Returns
    -------
    ind : int
    """
    sub = sub[::-1]
    ind = sub[0]
    sub = sub[1:]
    if sub:
        stride = shape[0]
        ind = ind + sub[0] * stride
        sub = sub[1:]
        shape = shape[1:]
        for i, s in zip(sub, shape):
            stride = stride * s
            ind = ind + i * stride
    return ind


@cuda.jit(device=True)
def ind2sub(ind: int, shape: List[int], sub: List[int]) -> List[int]:
    """Convert linear indices into sub indices (i, j, k).

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    ind : int
    shape : list[int]

    Returns
    -------
    sub : list[int]
    """
    for i in range(len(shape)-1):
        stride = 1
        for s in shape[i+1:]:
            stride *= s
        if i == 0:
            sub[i] = ind // stride
        else:
            stride_left = stride * shape[i]
            sub[i] = remainder(ind, stride_left) // stride
    return sub