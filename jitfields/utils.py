from typing import List, Sequence, TypeVar
from types import GeneratorType as generator
import math as pymath
import torch
import importlib
T = TypeVar('T')


def try_import(module, key=None):
    def try_import_module(path):
        try:
            return importlib.import_module(path)
        except (ImportError, ModuleNotFoundError):
            return None
    if key:
        fullmodule = try_import_module(module + '.' + key)
        if fullmodule:
            return fullmodule
    module = try_import_module(module)
    if not module:
        return None
    return getattr(module, key, None) if key else module


def remainder(x, d):
    return x - (x // d) * d


def prod(sequence: Sequence[T]) -> T:
    """Perform the cumulative product of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__mul__` operator is defined.
    Returns
    -------
    product : T
        Product of the elements in the sequence.

    """
    accumulate = None
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        else:
            accumulate = accumulate * elem
    return accumulate


def cumprod(sequence: Sequence[T],
            reverse: bool = False, exclusive: bool = False) -> List[T]:
    """Perform the cumulative product of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__mul__` operator is defined.
    reverse : bool, default=False
        Compute cumulative product from right-to-left:
        `cumprod([a, b, c], reverse=True) -> [a*b*c, b*c, c]`
    exclusive : bool, default=False
        Exclude self from the cumulative product:
        `cumprod([a, b, c], exclusive=True) -> [1, a, a*b]`

    Returns
    -------
    product : list
        Product of the elements in the sequence.

    """
    if reverse:
        sequence = reversed(sequence)
    accumulate = None
    seq = [1] if exclusive else []
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        else:
            accumulate = accumulate * elem
        seq.append(accumulate)
    if exclusive:
        seq = seq[:-1]
    if reverse:
        seq = list(reversed(seq))
    return seq


def cumsum(sequence: Sequence[T],
           reverse: bool = False, exclusive: bool = False) -> List[T]:
    """Perform the cumulative sum of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__sum__` operator is defined.
    reverse : bool, default=False
        Compute cumulative product from right-to-left:
        `cumprod([a, b, c], reverse=True) -> [a+b+c, b+c, c]`
    exclusive : bool, default=False
        Exclude self from the cumulative product:
        `cumprod([a, b, c], exclusive=True) -> [0, a, a+b]`

    Returns
    -------
    sum : list
        Sum of the elements in the sequence.

    """
    if reverse:
        sequence = reversed(sequence)
    accumulate = None
    seq = [0] if exclusive else []
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        else:
            accumulate = accumulate + elem
        seq.append(accumulate)
    if exclusive:
        seq = seq[:-1]
    if reverse:
        seq = list(reversed(seq))
    return seq


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
    *sub, ind = sub
    stride = cumprod(shape[1:], reverse=True)
    for i, s in zip(sub, stride):
        ind += i * s
    return ind


def ind2sub(ind: int, shape: List[int]) -> List[int]:
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
    stride = cumprod(shape, reverse=True, exclusive=True)
    sub: List[int] = []
    for s in stride:
        sub.append(int(remainder(ind, s)))
        ind = ind // s
    return sub


def ensure_list(x, size=None, crop=True, **kwargs):
    """Ensure that an object is a list (of size at last dim)

    If x is a list, nothing is done (no copy triggered).
    If it is a tuple, it is converted into a list.
    Otherwise, it is placed inside a list.
    """
    if not isinstance(x, (list, tuple, range, generator)):
        x = [x]
    elif not isinstance(x, list):
        x = list(x)
    if size and len(x) < size:
        default = kwargs.get('default', x[-1])
        x += [default] * (size - len(x))
    if size and crop:
        x = x[:size]
    return x


def make_vector(input, n=None, crop=True, *args,
                dtype=None, device=None, **kwargs):
    """Ensure that the input is a (tensor) vector and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    dtype : torch.dtype, optional
        Output data type.
    device : torch.device, optional
        Output device

    Returns
    -------
    output : tensor
        Output vector.

    """
    input = torch.as_tensor(input, dtype=dtype, device=device).flatten()
    if n is None:
        return input
    if n is not None and input.numel() >= n:
        return input[:n] if crop else input
    if args:
        default = args[0]
    elif 'default' in kwargs:
        default = kwargs['default']
    else:
        default = input[-1]
    default = input.new_full([n-len(input)], default)
    return torch.cat([input, default])
