import torch
from .utils import try_import, ensure_list, prod
from .splinc import spline_coeff_nd
import math as pymath
cuda_resize = try_import('jitfields.cuda', 'resize')
cpu_resize = try_import('jitfields.cpp', 'resize')
cuda_restrict = try_import('jitfields.cuda', 'restrict')
cpu_restrict = try_import('jitfields.cpp', 'restrict')
# from .cpp import restrict as cpu_restrict
# from .cpp import resize as cpu_resize


class _Resize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, factor, shape, ndim, anchor, order, bound, out):
        if x.is_cuda:
            _resize = cuda_resize.resize
        else:
            _resize = cpu_resize.resize
        ctx.opt = (x.shape, factor, shape, ndim, anchor, order, bound)
        x = _resize(x, factor, shape, ndim, anchor, order, bound, out)
        return x

    @staticmethod
    def backward(ctx, grad):
        inshape, factor, shape, ndim, anchor, order, bound = ctx.opt
        if grad.is_cuda:
            _restrict = cuda_restrict.restrict
        else:
            _restrict = cpu_restrict.restrict
        grad = _restrict(grad, factor, inshape[-ndim:], ndim, anchor, order, bound)
        return (grad,) + (None,) * 8


def resize(x, factor=None, shape=None, ndim=None,
           anchor='e', order=2, bound='dct2', prefilter=True, out=None):
    """Resize a tensor using spline interpolation

    Parameters
    ----------
    x : (..., *inshape) tensor
        Input  tensor
    factor : [sequence of] float, optional
        Factor by which to resize the tensor (> 1 == bigger)
        One of factor or shape must be provided.
    shape : [sequence of] float, optional
        Shape of output tensor.
        One of factor or shape must be provided.
    ndim : int, optional
        Number if spatial dimensions.
        If not provided, try to guess from factor or shape.
        If guess fails, assume ndim = x.dim().
    anchor : {'edge', 'center'} or None
        What feature should be aligned across the input and output tensors.
        If 'edge' or 'center', the effective scaling factor may slightly
        differ from the requested scaling factor.
        If None, the center of the (0, 0) voxel is aligned, and the
        requested factor is exactly applied.
    order : [sequence of] {0..7}, default=2
        Interpolation order.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
        How to deal with out-of-bound values.
    prefilter : bool, default=True
        Whether to first compute interpolating coefficients.
        Must be true for proper interpolation, otherwise this
        function merely performs a non-interpolating "prolongation".

    Returns
    -------
    x : (..., *shape) tensor
        Resized tensor

    """
    if not ndim:
        if shape and hasattr(shape, '__len__'):
            ndim = len(shape)
        elif factor and hasattr(factor, '__len__'):
            ndim = len(factor)
        else:
            ndim = x.dim()
    if shape:
        shape = ensure_list(shape, ndim)
    elif factor:
        factor = ensure_list(factor, ndim)
    else:
        raise ValueError('At least one of shape or factor must be provided')
    if not shape:
        if out is not None:
            shape = out.shape[-ndim:]
        else:
            shape = [pymath.ceil(s*f) for s, f in zip(x.shape[-ndim:], factor)]

    if prefilter:
        spline_coeff_nd(x, order, bound, ndim)
    return _Resize.apply(x, factor, shape, ndim, anchor, order, bound, out)


class _Restrict(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, factor, shape, ndim, anchor, order, bound, reduce_sum, out):
        if x.is_cuda:
            _restrict = cuda_restrict.restrict
        else:
            _restrict = cpu_restrict.restrict
        x, scale = _restrict(x, factor, shape, ndim, anchor, order, bound, out)
        scale = prod(scale)
        ctx.opt = (x.shape, factor, shape, ndim, anchor, order, bound, reduce_sum, scale)
        if not reduce_sum:
            x /= scale
        return x

    @staticmethod
    def backward(ctx, grad, *args):
        inshape, factor, shape, ndim, anchor, order, bound, reduce_sum, scale = ctx.opt
        if not reduce_sum:
            grad = grad / scale
        if grad.is_cuda:
            _resize = cuda_resize.resize
        else:
            _resize = cpu_resize.resize
        grad = _resize(grad, factor, inshape[-ndim:], ndim, anchor, order, bound)
        return (grad,) + (None,) * 8


def restrict(x, factor=None, shape=None, ndim=None,
             anchor='e', order=1, bound='dct2', reduce_sum=False, out=None):
    """Restrict (adjoint of resize) a tensor using spline interpolation

    Parameters
    ----------
    x : (..., *inshape) tensor
        Input  tensor
    factor : [sequence of] float, optional
        Factor by which to resize the tensor (> 1 == smaller)
        One of factor or shape must be provided.
    shape : [sequence of] float, optional
        Shape of output tensor.
        One of factor or shape must be provided.
    ndim : int, optional
        Number if spatial dimensions.
        If not provided, try to guess from factor or shape.
        If guess fails, assume ndim = x.dim().
    anchor : {'edge', 'center'} or None
        What feature should be aligned across the input and output tensors.
        If 'edge' or 'center', the effective scaling factor may slightly
        differ from the requested scaling factor.
        If None, the center of the (0, 0) voxel is aligned, and the
        requested factor is exactly applied.
    order : [sequence of] {0..7}, default=2
        Interpolation order.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
        How to deal with out-of-bound values.

    Returns
    -------
    x : (..., *shape) tensor
        restricted tensor

    """
    if not ndim:
        if shape and hasattr(shape, '__len__'):
            ndim = len(shape)
        elif factor and hasattr(factor, '__len__'):
            ndim = len(shape)
        else:
            ndim = x.dim()
    if shape:
        shape = ensure_list(shape, ndim)
    elif factor:
        factor = ensure_list(factor, ndim)
    else:
        raise ValueError('At least one of shape or factor must be provided')
    if not shape:
        if out is not None:
            shape = out.shape[-ndim:]
        else:
            shape = [pymath.ceil(s/f) for s, f in zip(x.shape[-ndim:], factor)]

    x = _Restrict.apply(x, factor, shape, ndim, anchor, order, bound, reduce_sum, out)
    return x

