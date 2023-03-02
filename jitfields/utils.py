from typing import List, Sequence, TypeVar
from types import GeneratorType as generator
import math as pymath
import torch
import importlib
import inspect
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
    default = input.new_full([], default).expand([n-len(input)])
    return torch.cat([input, default])


if 'indexing' in inspect.signature(torch.meshgrid).parameters:
    @torch.jit.script
    def meshgrid_script_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
        return torch.meshgrid(x, indexing='ij')
    @torch.jit.script
    def meshgrid_script_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
        return torch.meshgrid(x, indexing='xy')
    def meshgrid_ij(*x):
        return torch.meshgrid(*x, indexing='ij')
    def meshgrid_xy(*x):
        return torch.meshgrid(*x, indexing='xy')
else:
    @torch.jit.script
    def meshgrid_script_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
        return torch.meshgrid(x)
    @torch.jit.script
    def meshgrid_script_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
        grid = torch.meshgrid(x)
        if len(grid) > 1:
            grid[0] = grid[0].transpose(0, 1)
            grid[1] = grid[1].transpose(0, 1)
        return grid
    def meshgrid_ij(*x):
        return torch.meshgrid(*x)
    def meshgrid_xy(*x):
        grid = list(torch.meshgrid(*x))
        if len(grid) > 1:
            grid[0] = grid[0].transpose(0, 1)
            grid[1] = grid[1].transpose(0, 1)
        return grid


def identity_grid(shape, dtype=None, device=None):
    """Returns an identity deformation field.

    Parameters
    ----------
    shape : (dim,) sequence of int
        Spatial dimension of the field.
    dtype : torch.dtype, default=`get_default_dtype()`
        Data type.
    device torch.device, optional
        Device.

    Returns
    -------
    grid : (*shape, dim) tensor
        Transformation field

    """
    mesh1d = [torch.arange(float(s), dtype=dtype, device=device)
              for s in shape]
    grid = meshgrid_ij(*mesh1d)
    grid = torch.stack(grid, dim=-1)
    return grid


@torch.jit.script
def _movedim1(x, source: int, destination: int):
    dim = x.dim()
    source = dim + source if source < 0 else source
    destination = dim + destination if destination < 0 else destination
    permutation = [d for d in range(dim)]
    permutation = permutation[:source] + permutation[source+1:]
    permutation = permutation[:destination] + [source] + permutation[destination:]
    return x.permute(permutation)


@torch.jit.script
def add_identity_grid_(disp):
    """Adds the identity grid to a displacement field, inplace.

    Parameters
    ----------
    disp : (..., *spatial, dim) tensor
        Displacement field

    Returns
    -------
    grid : (..., *spatial, dim) tensor
        Transformation field

    """
    dim = disp.shape[-1]
    spatial = disp.shape[-dim-1:-1]
    mesh1d = [torch.arange(s, dtype=disp.dtype, device=disp.device)
              for s in spatial]
    grid = meshgrid_script_ij(mesh1d)
    disp = _movedim1(disp, -1, 0)
    for i, grid1 in enumerate(grid):
        disp[i].add_(grid1)
    disp = _movedim1(disp, 0, -1)
    return disp


@torch.jit.script
def add_identity_grid(disp):
    """Adds the identity grid to a displacement field.

    Parameters
    ----------
    disp : (..., *spatial, dim) tensor
        Displacement field

    Returns
    -------
    grid : (..., *spatial, dim) tensor
        Transformation field

    """
    return add_identity_grid_(disp.clone())


def affine_grid(mat, shape):
    """Create a dense transformation grid from an affine matrix.

    Parameters
    ----------
    mat : (..., D[+1], D[+1]) tensor
        Affine matrix (or matrices).
    shape : (D,) sequence[int]
        Shape of the grid, with length D.

    Returns
    -------
    grid : (..., *shape, D) tensor
        Dense transformation grid

    """
    mat = torch.as_tensor(mat)
    shape = list(shape)
    nb_dim = mat.shape[-1] - 1
    if nb_dim != len(shape):
        raise ValueError('Dimension of the affine matrix ({}) and shape ({}) '
                         'are not the same.'.format(nb_dim, len(shape)))
    if mat.shape[-2] not in (nb_dim, nb_dim+1):
        raise ValueError('First argument should be matrces of shape '
                         '(..., {0}, {1}) or (..., {1], {1}) but got {2}.'
                         .format(nb_dim, nb_dim+1, mat.shape))
    batch_shape = mat.shape[:-2]
    grid = identity_grid(shape, mat.dtype, mat.device)
    if batch_shape:
        for _ in range(len(batch_shape)):
            grid = grid[None]
        for _ in range(nb_dim):
            mat = mat[..., None, :, :]
    lin = mat[..., :nb_dim, :nb_dim]
    off = mat[..., :nb_dim, -1]
    grid = lin.matmul(grid.unsqueeze(-1)).squeeze(-1) + off
    return grid


def broadcast(x, y, skip_last=0):
    """Broadcast the shapes of two tensors.

    The last `skip_last` dimensions are preserved and not included in
    the broadcast.

    Parameters
    ----------
    x : tensor
    y : tensor
    skip_last : int or (int, int), default=0

    Returns
    -------
    x : tensor
    y : tensor

    """
    ndim = max(x.dim(), y.dim())
    while x.dim() < ndim:
        x = x[None]
    while y.dim() < ndim:
        y = y[None]
    xskip, yskip = ensure_list(skip_last, 2)
    xslicer = slice(-xskip if xskip else None)
    yslicer = slice(-yskip if yskip else None)
    xbatch = x.shape[xslicer]
    ybatch = y.shape[yslicer]
    batch = []
    for bx, by, in zip(xbatch, ybatch):
        if bx > 1 and by > 1 and bx != by:
            raise ValueError('Cannot broadcast batch shapes',
                             tuple(xbatch), 'and', tuple(ybatch))
        batch.append(max(bx, by))
    xslicer = slice(-xskip if xskip else None, None)
    yslicer = slice(-yskip if yskip else None, None)
    xshape = batch + (list(x.shape[xslicer]) if xskip else [])
    yshape = batch + (list(y.shape[yslicer]) if yskip else [])
    x = x.expand(xshape)
    y = y.expand(yshape)
    return x, y
