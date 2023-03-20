__all__ = ['resize', 'restrict']

import torch
from .bindings.common.bounds import convert_bound
from .bindings.common.spline import convert_order
from .utils import try_import, ensure_list, prod
from .splinc import spline_coeff_nd
import math as pymath
cuda_resize = try_import('jitfields.bindings.cuda', 'resize')
cpu_resize = try_import('jitfields.bindings.cpp', 'resize')
cuda_restrict = try_import('jitfields.bindings.cuda', 'restrict')
cpu_restrict = try_import('jitfields.bindings.cpp', 'restrict')


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
    if factor:
        factor = ensure_list(factor, ndim)
    if not shape and not factor:
        raise ValueError('At least one of shape or factor must be provided')

    if not shape:
        if out is not None:
            shape = out.shape[-ndim:]
        else:
            shape = [pymath.ceil(s*f) for s, f in zip(x.shape[-ndim:], factor)]

    fullshape = list(x.shape[:-ndim]) + list(shape)
    if out is None:
        out = x.new_empty(fullshape)
    else:
        out = out.expand(fullshape)

    order = [convert_order.get(o, o) for o in ensure_list(order, ndim)]
    bound = [convert_bound.get(b, b) for b in ensure_list(bound, ndim)]
    if prefilter:
        x = spline_coeff_nd(x, order, bound, ndim)
    return _Resize.apply(x, factor, shape, ndim, anchor, order, bound, out)


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

    fullshape = list(x.shape[:-ndim]) + list(shape)
    if out is None:
        out = x.new_empty(fullshape)
    else:
        out = out.expand(fullshape)

    order = [convert_order.get(o, o) for o in ensure_list(order, ndim)]
    bound = [convert_bound.get(b, b) for b in ensure_list(bound, ndim)]
    x = _Restrict.apply(x, factor, shape, ndim, anchor, order, bound, reduce_sum, out)
    return x


class _Resize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, factor, shape, ndim, anchor, order, bound, out):
        resize = (cuda_resize if x.is_cuda else cpu_resize).resize
        ctx.opt = (x.shape, factor, shape, ndim, anchor, order, bound)
        x = resize(out, x, factor, anchor, order, bound)
        return x

    @staticmethod
    def backward(ctx, grad):
        inshape, factor, shape, ndim, anchor, order, bound = ctx.opt
        restrict = (cuda_restrict if grad.is_cuda else cpu_restrict).restrict
        out = grad.new_empty([*grad.shape[:-ndim], *inshape[-ndim:]])
        restrict(out, grad, factor, anchor, order, bound)
        return (out,) + (None,) * 7


class _Restrict(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, factor, shape, ndim, anchor, order, bound, reduce_sum, out):
        restrict = (cuda_restrict if x.is_cuda else cpu_restrict).restrict
        x, scale = restrict(out, x, factor, anchor, order, bound)
        scale = prod(scale)
        ctx.opt = (x.shape, factor, shape, ndim, anchor, order, bound, reduce_sum, scale)
        if not reduce_sum:
            x /= scale
        return x

    @staticmethod
    def backward(ctx, grad, *args):
        inshape, factor, shape, ndim, anchor, order, bound, reduce_sum, scale = ctx.opt
        resize = (cuda_resize if grad.is_cuda else cpu_resize).resize
        if not reduce_sum:
            grad = grad / scale
        out = grad.new_empty([*grad.shape[:-ndim], *inshape[-ndim:]])
        out = resize(out, grad, factor, anchor, order, bound)
        return (out,) + (None,) * 8
