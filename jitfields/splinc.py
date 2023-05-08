__all__ = [
    'spline_coeff', 'spline_coeff_',
    'spline_coeff_nd', 'spline_coeff_nd_'
]

import torch.autograd
from torch import Tensor
from typing import Optional
from .bindings.common.bounds import convert_bound, cnames as boundnames
from .bindings.common.spline import convert_order
from .utils import try_import, ensure_list
from .typing import OneOrSeveral, BoundType, OrderType

cuda_splinc = try_import('jitfields.bindings.cuda', 'splinc')
cpu_splinc = try_import('jitfields.bindings.cpp', 'splinc')


def spline_coeff(
    inp: Tensor,
    order: OrderType,
    bound: BoundType = 'dct2',
    dim: int = -1,
) -> Tensor:
    """Compute the interpolating spline coefficients, along a single dimension.

    Parameters
    ----------
    inp : `tensor`
        Input tensor
    order : `{0..7}`, default=2
        Interpolation order.
    bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dct2'
        Boundary conditions.
    dim : `int`, default=-1
        Dimension along which to filter

    Returns
    -------
    coeff : `tensor`
        Spline coefficients
    """
    return spline_coeff_(inp.clone(), order, bound, dim)


def spline_coeff_(
    inp: Tensor,
    order: OrderType,
    bound: BoundType = 'dct2',
    dim: int = -1,
) -> Tensor:
    """Compute the interpolating spline coefficients, along a single dimension.

    Notes
    -----
    This function operates inplace.

    Parameters
    ----------
    inp : `tensor`
        Input tensor
    order : `{0..7}`, default=2
        Interpolation order.
    bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dct2'
        Boundary conditions.
    dim : `int`, default=-1
        Dimension along which to filter

    Returns
    -------
    coeff : `tensor`
        Spline coefficients
    """
    order = convert_order.get(order, order)
    bound = convert_bound.get(bound, bound)
    checkbound(order, bound)
    return SplineCoeff_.apply(inp, order, bound, dim)


def spline_coeff_nd(
    inp: Tensor,
    order: OneOrSeveral[OrderType],
    bound: OneOrSeveral[BoundType] = 'dct2',
    ndim: Optional[int] = None,
) -> Tensor:
    """Compute the interpolating spline coefficients, along the last N dimensions.

    Parameters
    ----------
    inp : `(..., *spatial) tensor`
        Input tensor, with shape `(..., *spatial)`.
    order : `[sequence of] {0..7}`, default=2
        Interpolation order (per dimension).
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dct2'
        Boundary conditions (per dimension).
    ndim : `int`, default=`inp.dim()`
        Number of spatial dimensions. Defaults: all.

    Returns
    -------
    coeff : `(..., *spatial) tensor`
        Spline coefficients, with shape `(..., *spatial)`.
    """
    return spline_coeff_nd_(inp.clone(), order, bound, ndim)


def spline_coeff_nd_(
    inp: Tensor,
    order: OneOrSeveral[OrderType],
    bound: OneOrSeveral[BoundType] = 'dct2',
    ndim: Optional[int] = None,
) -> Tensor:
    """Compute the interpolating spline coefficients, along the last N dimensions.

    Notes
    -----
    This function operates inplace.

    Parameters
    ----------
    inp : `(..., *spatial) tensor`
        Input tensor, with shape `(..., *spatial)`.
    order : `[sequence of] {0..7}`, default=2
        Interpolation order (per dimension).
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dct2'
        Boundary conditions (per dimension).
    ndim : `int`, default=`inp.dim()`
        Number of spatial dimensions. Defaults: all.

    Returns
    -------
    coeff : `(..., *spatial) tensor`
        Spline coefficients, with shape `(..., *spatial)`.
    """
    order = [convert_order.get(o, o) for o in ensure_list(order, ndim)]
    bound = [convert_bound.get(b, b) for b in ensure_list(bound, ndim)]
    checkbound(order, bound)
    return SplineCoeffND_.apply(inp, order, bound, ndim)


bounds_ok = ('dct1', 'dct2', 'dft', 'replicate')
bounds_ok_int = list(map(lambda x: convert_bound[x], bounds_ok))


def checkbound(order, bound):
    if any(o != 1 and b not in bounds_ok_int for o, b in zip(order, bound)):
        bound = tuple(boundnames[b].lower() for b in bound)
        raise ValueError(f'`spline_coeff` only implemented for bounds '
                         f'{bounds_ok} but got: {bound}')


class SplineCoeff_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, order, bound, dim):
        if inp.is_cuda:
            spline_coeff_ = cuda_splinc.spline_coeff_
        else:
            spline_coeff_ = cpu_splinc.spline_coeff_
        ctx.opt = (order, bound, dim)
        return spline_coeff_(inp, order, bound, dim)

    @staticmethod
    def backward(ctx, grad):
        if grad.is_cuda:
            spline_coeff_ = cuda_splinc.spline_coeff_
        else:
            spline_coeff_ = cpu_splinc.spline_coeff_
        grad = spline_coeff_(grad.clone(), *ctx.opt)
        return (grad,) + (None,) * 3


class SplineCoeffND_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, order, bound, ndim):
        if inp.is_cuda:
            spline_coeff_nd_ = cuda_splinc.spline_coeff_nd_
        else:
            spline_coeff_nd_ = cpu_splinc.spline_coeff_nd_
        ctx.opt = (order, bound, ndim)
        return spline_coeff_nd_(inp, order, bound, ndim)

    @staticmethod
    def backward(ctx, grad):
        if grad.is_cuda:
            spline_coeff_nd_ = cuda_splinc.spline_coeff_nd_
        else:
            spline_coeff_nd_ = cpu_splinc.spline_coeff_nd_
        grad = spline_coeff_nd_(grad.clone(), *ctx.opt)
        return (grad,) + (None,) * 3
