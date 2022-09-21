from .utils import try_import
cuda_splinc = try_import('jitfields.cuda', 'splinc')
cpu_splinc = try_import('jitfields.cpp', 'splinc')
from jitfields.cpp import splinc as cpu_splinc


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
    if inp.is_cuda:
        spline_coeff_ = cuda_splinc.spline_coeff_
    else:
        spline_coeff_ = cpu_splinc.spline_coeff_
    return spline_coeff_(inp, order, bound, dim)


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
    if inp.is_cuda:
        spline_coeff_nd_ = cuda_splinc.spline_coeff_nd_
    else:
        spline_coeff_nd_ = cpu_splinc.spline_coeff_nd_
    return spline_coeff_nd_(inp, order, bound, ndim)


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
