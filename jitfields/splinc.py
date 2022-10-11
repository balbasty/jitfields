import torch.autograd
from .common.bounds import convert_bound
from .common.spline import convert_order
from .utils import try_import, ensure_list
cuda_splinc = try_import('jitfields.cuda', 'splinc')
cpu_splinc = try_import('jitfields.cpp', 'splinc')


def spline_coeff_(inp, order, bound='dct2', dim=-1):
    """Compute the interpolating spline coefficients, along a single dimension.

    Parameters
    ----------
    inp : tensor
        Input tensor
    order : {0..7}, default=2
        Interpolation order.
    bound : {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
        Boundary conditions.
    dim : int, default=-1
        Dimension along which to filter

    Returns
    -------
    coeff : tensor
        Spline coefficients
    """
    order = convert_order.get(order, order)
    bound = convert_bound.get(bound, bound)
    return SplineCoeff_.apply(inp, order, bound, dim)


def spline_coeff(inp, order, bound='dct2', dim=-1):
    """Compute the interpolating spline coefficients, along a single dimension.

    Parameters
    ----------
    inp : tensor
        Input tensor
    order : {0..7}, default=2
        Interpolation order.
    bound : {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
        Boundary conditions.
    dim : int, default=-1
        Dimension along which to filter

    Returns
    -------
    coeff : tensor
        Spline coefficients
    """
    return spline_coeff_(inp.clone(), order, bound, dim)


def spline_coeff_nd_(inp, order, bound='dct2', ndim=None):
    """Compute the interpolating spline coefficients, along the last N dimensions.

    Parameters
    ----------
    inp : (..., *spatial) tensor
        Input tensor
    order : [sequence of] {0..7}, default=2
        Interpolation order.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
        Boundary conditions.
    ndim : int, default=`inp.dim()`
        Number of spatial dimensions

    Returns
    -------
    coeff : (..., *spatial) tensor
        Spline coefficients
    """
    order = [convert_order.get(o, o) for o in ensure_list(order, ndim)]
    bound = [convert_bound.get(b, b) for b in ensure_list(bound, ndim)]
    return SplineCoeffND_.apply(inp, order, bound, ndim)


def spline_coeff_nd(inp, order, bound='dct2', ndim=None):
    """Compute the interpolating spline coefficients, along the last N dimensions.

    Parameters
    ----------
    inp : (..., *spatial) tensor
        Input tensor
    order : [sequence of] {0..7}, default=2
        Interpolation order.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
        Boundary conditions.
    ndim : int, default=`inp.dim()`
        Number of spatial dimensions

    Returns
    -------
    coeff : (..., *spatial) tensor
        Spline coefficients
    """
    return spline_coeff_nd_(inp.clone(), order, bound, ndim)


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
        return spline_coeff_(grad.clone(), *ctx.opt)


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
        return spline_coeff_nd_(grad.clone(), *ctx.opt)
