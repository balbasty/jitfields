"""
Semi-hidden module that exposes the (non-differentiable) backward passes
of push/pull.
"""
__all__ = ['pull_backward', 'push_backward', 'count_backward', 'grad_backward']

import torch
from torch import Tensor
from typing import Tuple
from .utils import try_import, ensure_list
from .typing import OneOrSeveral, BoundType, OrderType, ExtrapolateType
from .splinc import spline_coeff_nd
from .bindings.common.bounds import convert_bound
from .bindings.common.spline import convert_order
cuda_pushpull = try_import('jitfields.bindings.cuda', 'pushpull')
cpu_pushpull = try_import('jitfields.bindings.cpp', 'pushpull')


def pull_backward(
    grad: Tensor,
    inp: Tensor,
    grid: Tensor,
    order: OneOrSeveral[OrderType] = 2,
    bound: OneOrSeveral[BoundType] = 'dct2',
    extrapolate: ExtrapolateType = True,
    prefilter: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Sample a tensor using spline interpolation

    !!! note
        By default, `inp` is assumed to contain the coefficients of a
        continuous function encoded by regularly spaced cubic splines.
        Instead, when `prefilter` is `True`, `pull` interpolates the values
        of `inp`. To this end, `inp` is first converted to spline coefficients
        (_i.e._, prefiltered).

    !!! warning
        Backpropagation does not work through this function!

    Parameters
    ----------
    grad : `(..., *outshape, channel) tensor`
        Gradient with respect to the output of `pull`,
        with shape `(..., *outshape, channel)`.
    inp : `(..., *inshape, channel) tensor`
        Input tensor with shape `(..., *inshape, channel)`.
    grid : `(..., *outshape, ndim) tensor`
        Tensor of coordinates into `inp`, with shape `(..., *outshape, ndim)`.
    order : `[sequence of] {0..7}`, default=2
        Interpolation order (per dimension).
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dct2'
        How to deal with out-of-bound values (per dimension).
    extrapolate : `bool or {'center', 'edge'}`
        - `True`: use bound to extrapolate out-of-bound value
        - `False` or `'center'`: do not extrapolate values that fall outside
          of the centers of the first and last voxels.
        - `'edge'`: do not extrapolate values that fall outside
           of the edges of the first and last voxels.
    prefilter : `bool, default=True
        Whether to first compute interpolating coefficients.
        Must be true for proper interpolation, otherwise this
        function merely performs a non-interpolating "spline sampling".

    Returns
    -------
    grad_inp : `(..., *inshape, channel) tensor`
        Gradient with respect to the input tensor,
         with shape `(..., *inshape, channel)`.
    grad_grid : `(..., *outshape, ndim) tensor`
        Gradient with respect to the tensor of coordinates into `inp`,
        with shape `(..., *outshape, ndim)`.

    """
    ndim = grid.shape[-1]
    if ndim > 3:
        raise NotImplementedError("Not implemented for spatial dim > 3")
    if prefilter:
        inp = spline_coeff_nd(inp.movedim(-1, 0), order, bound, ndim).movedim(0, -1)
    inp, grid = _broadcast_pull(inp, grid)
    order, bound, extrapolate = _preproc_opt(order, bound, extrapolate, ndim)

    bwd = (cuda_pushpull if grad.is_cuda else cpu_pushpull).pull_backward
    outgrad_inp = torch.zeros_like(inp)
    outgrad_grid = torch.empty_like(grid)
    bwd(outgrad_inp, outgrad_grid, grad, inp, grid, order, bound, extrapolate)
    return outgrad_inp, outgrad_grid


def grad_backward(
    grad: Tensor,
    inp: Tensor,
    grid: Tensor,
    order: OneOrSeveral[OrderType] = 2,
    bound: OneOrSeveral[BoundType] = 'dct2',
    extrapolate: ExtrapolateType = True,
    prefilter: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Sample the spatial gradients of a tensor using spline interpolation

    !!! note
        By default, `inp` is assumed to contain the coefficients of a
        continuous function encoded by regularly spaced cubic splines.
        Instead, when `prefilter` is `True`, `grad` interpolates the values
        of `inp`. To this end, `inp` is first converted to spline coefficients
        (_i.e._, prefiltered).

    !!! warning
        Backpropagation does not work through this function!

    Parameters
    ----------
    grad : `(..., *outshape, channel, ndim) tensor`
        Gradient with respect to the output of `grad`,
        with shape `(..., *outshape, channel, ndim)`.
    inp : `(..., *inshape, channel) tensor`
        Input tensor, with shape `(..., *inshape, channel)`.
    grid : `(..., *outshape, ndim) tensor`
        Tensor of coordinates into `inp`, with shape `(..., *outshape, ndim)`.
    order : [sequence of] {0..7}, default=2
        Interpolation order (per dimension).
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dct2'
        How to deal with out-of-bound values (per dimension).
    extrapolate : `bool or {'center', 'edge'}`
        - `True`: use bound to extrapolate out-of-bound value
        - `False` or `'center'`: do not extrapolate values that fall outside
          of the centers of the first and last voxels.
        - `'edge'`: do not extrapolate values that fall outside
           of the edges of the first and last voxels.
    prefilter : `bool`, default=True
        Whether to first compute interpolating coefficients.
        Must be true for proper interpolation, otherwise this
        function merely performs a non-interpolating "spline sampling".

    Returns
    -------
    grad_inp : `(..., *inshape, channel) tensor`
        Gradient with respect to the input tensor,
         with shape `(..., *inshape, channel)`.
    grad_grid : `(..., *outshape, ndim) tensor`
        Gradient with respect to the tensor of coordinates into `inp`,
        with shape `(..., *outshape, ndim)`.

    """
    ndim = grid.shape[-1]
    if ndim > 3:
        raise NotImplementedError("Not implemented for spatial dim > 3")
    if prefilter:
        inp = spline_coeff_nd(inp.movedim(-1, 0), order, bound, ndim).movedim(0, -1)
    inp, grid = _broadcast_pull(inp, grid)
    order, bound, extrapolate = _preproc_opt(order, bound, extrapolate, ndim)

    bwd = (cuda_pushpull if grad.is_cuda else cpu_pushpull).grad_backward
    outgrad_inp = torch.zeros_like(inp)
    outgrad_grid = torch.empty_like(grid)
    bwd(outgrad_inp, outgrad_grid, grad, inp, grid, order, bound, extrapolate)
    return outgrad_inp, outgrad_grid


def push_backward(
    grad: Tensor,
    inp: Tensor,
    grid: Tensor,
    order: OneOrSeveral[OrderType] = 2,
    bound: OneOrSeveral[BoundType] = 'dct2',
    extrapolate: ExtrapolateType = True,
    prefilter: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Splat a tensor using spline interpolation

    !!! warning
        Backpropagation does not work through this function!

    Parameters
    ----------
    grad : `(..., *outshape, channel) tensor`
        Gradient with respect to the output of `push`,
        with shape `(..., *outshape, channel)`.
    inp : `(..., *inshape, channel) tensor`
        Input tensor, with shape `(..., *inshape, channel)`.
    grid : `(..., *inshape, ndim) tensor`
        Tensor of coordinates into `inp`, with shape `(..., *inshape, ndim)`.
    order : `[sequence of] {0..7}`, default=2
        Interpolation order (per dimension).
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dct2'
        How to deal with out-of-bound values (per dimension).
    extrapolate : `bool or {'center', 'edge'}`
        - `True`: use bound to extrapolate out-of-bound value
        - `False` or `'center'`: do not extrapolate values that fall outside
          of the centers of the first and last voxels.
        - `'edge'`: do not extrapolate values that fall outside
           of the edges of the first and last voxels.
    prefilter : `bool`, default=True
        Whether to compute interpolating coefficients at the end.
        If the value for `prefilter` is matched across `pull` and `push`,
        they are adjoint of each other.

    Returns
    -------
    grad_inp : `(..., *inshape, channel) tensor`
        Gradient with respect to the input tensor,
         with shape `(..., *inshape, channel)`.
    grad_grid : `(..., *inshape, ndim) tensor`
        Gradient with respect to the tensor of coordinates into `inp`,
        with shape `(..., *inshape, ndim)`.

    """
    ndim = grid.shape[-1]
    if ndim > 3:
        raise NotImplementedError("Not implemented for spatial dim > 3")
    inp, grid = _broadcast_push(inp, grid)
    order, bound, extrapolate = _preproc_opt(order, bound, extrapolate, ndim)

    if prefilter:
        grad = grad.movedim(-1, 0)
        grad = spline_coeff_nd(grad, order, bound, ndim)
        grad = grad.movedim(0, -1)

    bwd = (cuda_pushpull if grad.is_cuda else cpu_pushpull).push_backward
    outgrad_inp = torch.empty_like(inp)
    outgrad_grid = torch.empty_like(grid)
    bwd(outgrad_inp, outgrad_grid, grad, inp, grid, order, bound, extrapolate)
    return outgrad_inp, outgrad_grid


def count_backward(
    grad: Tensor,
    grid: Tensor,
    order: OneOrSeveral[OrderType] = 2,
    bound: OneOrSeveral[BoundType] = 'dct2',
    extrapolate: ExtrapolateType = True,
) -> Tensor:
    """Splat ones using spline interpolation

    !!! warning
        Backpropagation does not work through this function!

    Parameters
    ----------
    grad : `(..., *outshape) tensor`
        Gradient with respect to the output of `count`,
        with shape `(..., *outshape, channel)`.
    grid : `(..., *inshape, ndim) tensor`
        Tensor of coordinates, with shape `(..., *inshape, ndim)`
    order : `[sequence of] {0..7}`, default=2
        Interpolation order (per dimension).
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dct2'
        How to deal with out-of-bound values (per dimension).
    extrapolate : `bool or {'center', 'edge'}`
        - `True`: use bound to extrapolate out-of-bound value
        - `False` or `'center'`: do not extrapolate values that fall outside
          of the centers of the first and last voxels.
        - `'edge'`: do not extrapolate values that fall outside
           of the edges of the first and last voxels.

    Returns
    -------
    grad_grid : `(..., *inshape, ndim) tensor`
        Gradient with respect to the tensor of coordinates,
        with shape `(..., *inshape, ndim)`.

    """
    ndim = grid.shape[-1]
    if ndim > 3:
        raise NotImplementedError("Not implemented for spatial dim > 3")
    order, bound, extrapolate = _preproc_opt(order, bound, extrapolate, ndim)

    bwd = (cuda_pushpull if grad.is_cuda else cpu_pushpull).count_backward
    outgrad_grid = torch.empty_like(grid)
    bwd(outgrad_grid, grad.unsqueeze(-1), grid, order, bound, extrapolate)
    return outgrad_grid


def _preproc_opt(order, bound, extrapolate, ndim):
    order = [convert_order.get(o, o) for o in ensure_list(order, ndim)]
    bound = [convert_bound.get(b, b) for b in ensure_list(bound, ndim)]
    extrapolate = _extrapolate(extrapolate)
    return order, bound, extrapolate


def _extrapolate(extrapolate):
    if isinstance(extrapolate, str):
        extrapolate = extrapolate[0].lower()
        extrapolate = -1 if extrapolate == 'e' else 0
    extrapolate = int(extrapolate)
    return extrapolate


def _broadcast(x, g, skip_last=0):
    ndim = max(x.dim(), g.dim())
    while x.dim() < ndim:
        x = x[None]
    while g.dim() < ndim:
        g = g[None]
    slicer = slice(-skip_last if skip_last else None)
    xbatch = x.shape[slicer]
    gbatch = g.shape[slicer]
    batch = []
    for bx, bg, in zip(xbatch, gbatch):
        if bx > 1 and bg > 1 and bx != bg:
            raise ValueError('Cannot broadcast batch shapes', tuple(xbatch),
                             'and', tuple(gbatch))
        batch.append(max(bx, bg))
    if skip_last:
        slicer = slice(-skip_last, None)
        x = x.expand(batch + list(x.shape[slicer]))
        g = g.expand(batch + list(g.shape[slicer]))
    else:
        x = x.expand(batch)
        g = g.expand(batch)
    return x, g


def _broadcast_pull(x, g):
    return _broadcast(x, g, skip_last=g.shape[-1] + 1)


def _broadcast_push(x, g):
    return _broadcast(x, g, skip_last=1)
