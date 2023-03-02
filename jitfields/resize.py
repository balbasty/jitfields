__all__ = ['resize', 'restrict',
           'Prolong', 'Restrict', 'GridProlong', 'GridRestrict']

import torch
from .utils import try_import, ensure_list, prod
from .splinc import spline_coeff_nd
import math as pymath
cuda_resize = try_import('jitfields.cuda', 'resize')
cpu_resize = try_import('jitfields.cpp', 'resize')
cuda_restrict = try_import('jitfields.cuda', 'restrict')
cpu_restrict = try_import('jitfields.cpp', 'restrict')


class Prolong:
    """Prolongation operator used in multi-grid solvers"""

    def __init__(self, ndim, factor=2, order=2, bound='dct2', anchor='e',
                 channel_last=False):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        factor : [list of] int
            Prolongation factor
        order : [list of] int
            Interplation order
        bound : [list of] str
            Boundary conditions
        anchor : [list of] {'edge', 'center'}
            Anchor points
        channel_last : bool, default=False
            Whether the channel dimension is last
        """
        self.ndim = ndim
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor
        self.channel_last = channel_last

    def __call__(self, x, out=None):
        """
        Parameters
        ----------
        inp : (..., *spatial_in, [channel]) tensor
            Tensor to prolongate
        out : (..., *spatial_out, [channel]) tensor, optional
            Output placeholder

        Returns
        -------
        out : (..., *spatial_out, [channel]) tensor, optional
            Prolongated tensor
        """
        if self.channel_last:
            x = torch.movedim(x, -1, -self.ndim-1)
            if out is not None:
                out = torch.movedim(out, -1, -self.ndim-1)
        if out is not None:
            prm = dict(shape=out.shape[-self.ndim:])
        else:
            prm = dict(factor=self.factor)
        out = resize(x, **prm, ndim=self.ndim,
                     order=self.order, bound=self.bound, anchor=self.anchor,
                     prefilter=False, out=out)
        if self.channel_last:
            out = torch.movedim(out, -self.ndim - 1, -1)
        return out


class GridProlong:
    """Prolongation operator for displacement fields used in multi-grid solvers"""

    def __init__(self, ndim, factor=2, order=2, bound='dft', anchor='e'):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        factor : [list of] int
            Prolongation factor
        order : [list of] int
            Interplation order
        bound : [list of] str
            Boundary conditions
        anchor : [list of] {'edge', 'center'}
            Anchor points
        """
        self.ndim = ndim
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor

    def get_scale(self, inshape, outshape):
        anchor = self.anchor[0].lower()
        factor = ensure_list(self.factor, self.ndim)
        if anchor == 'e':
            scale = [so / si for si, so in zip(inshape, outshape)]
        elif anchor == 'c':
            scale = [(so - 1) / (si - 1) for si, so in zip(inshape, outshape)]
        else:
            scale = factor
        return scale

    def __call__(self, x, out=None):
        """
        Parameters
        ----------
        inp : (..., *spatial_in, ndim) tensor
            Tensor to prolongate
        out : (..., *spatial_out, ndim) tensor, optional
            Output placeholder

        Returns
        -------
        out : (..., *spatial_out, ndim) tensor, optional
            Prolongated tensor
        """
        ndim = self.ndim
        if out is not None:
            prm = dict(shape=out.shape[-ndim-1:-1])
        else:
            prm = dict(factor=self.factor)
        x = torch.movedim(x, -1, -ndim-1)
        if out is not None:
            out = torch.movedim(out, -1, -ndim-1)
        out = resize(x, **prm, ndim=ndim,
                     order=self.order, bound=self.bound, anchor=self.anchor,
                     prefilter=False, out=out)
        scale = self.get_scale(x.shape[-ndim:], out.shape[-ndim:])
        out = torch.movedim(out, -ndim-1, -1)
        if out.shape[-1] == ndim:
            # Gradient
            for d, out1 in enumerate(out[..., :ndim].unbind(-1)):
                out1 *= scale[d]
        else:
            # if Hessian
            c = ndim
            for d in range(ndim):
                out[..., d] *= scale[d] * scale[d]
                for dd in range(d+1, ndim):
                    out[..., c] *= scale[d] * scale[dd]
                    c += 1
        return out


class Restrict:
    """Restriction operator used in multi-grid solvers"""

    def __init__(self, ndim, factor=2, order=1, bound='dct2', anchor='e',
                 channel_last=False):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        factor : [list of] int
            Prolongation factor
        order : [list of] int
            Interplation order
        bound : [list of] str
            Boundary conditions
        anchor : [list of] {'edge', 'center'}
            Anchor points
        channel_last : bool, default=False
            Whether the channel dimension is last
        """
        self.ndim = ndim
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor
        self.channel_last = channel_last

    def __call__(self, x, out=None):
        """
        Parameters
        ----------
        inp : (..., *spatial_in, [channel]) tensor
            Tensor to prolongate
        out : (..., *spatial_out, [channel]) tensor, optional
            Output placeholder

        Returns
        -------
        out : (..., *spatial_out, [channel]) tensor, optional
            Prolongated tensor
        """
        if self.channel_last:
            x = torch.movedim(x, -1, -self.ndim-1)
            if out is not None:
                out = torch.movedim(out, -1, -self.ndim-1)
        if out is not None:
            prm = dict(shape=out.shape[-self.ndim:])
        else:
            prm = dict(factor=self.factor)
        out = restrict(x, **prm, ndim=self.ndim,
                       order=self.order, bound=self.bound, anchor=self.anchor,
                       out=out)
        if self.channel_last:
            out = torch.movedim(out, -self.ndim - 1, -1)
        return out


class GridRestrict:
    """Restriction operator for displacement fields used in multi-grid solvers"""

    def __init__(self, ndim, factor=2, order=1, bound='dft', anchor='e'):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        factor : [list of] int
            Prolongation factor
        order : [list of] int
            Interplation order
        bound : [list of] str
            Boundary conditions
        anchor : [list of] {'edge', 'center'}
            Anchor points
        """
        self.ndim = ndim
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor

    def get_scale(self, inshape, outshape):
        anchor = self.anchor[0].lower()
        factor = ensure_list(self.factor, self.ndim)
        if anchor == 'e':
            scale = [so / si for si, so in zip(inshape, outshape)]
        elif anchor == 'c':
            scale = [(so - 1) / (si - 1) for si, so in zip(inshape, outshape)]
        else:
            scale = [1 / f for f in factor]
        return scale

    def __call__(self, x, out=None):
        """
        Parameters
        ----------
        inp : (..., *spatial_in, ndim) tensor
            Tensor to prolongate
        out : (..., *spatial_out, ndim) tensor, optional
            Output placeholder

        Returns
        -------
        out : (..., *spatial_out, ndim) tensor, optional
            Prolongated tensor
        """
        ndim = self.ndim
        if out is not None:
            prm = dict(shape=out.shape[-ndim-1:-1])
        else:
            prm = dict(factor=self.factor)
        x = torch.movedim(x, -1, -ndim-1)
        if out is not None:
            out = torch.movedim(out, -1, -ndim-1)
        out = restrict(x, **prm, ndim=ndim,
                       order=self.order, bound=self.bound, anchor=self.anchor,
                       out=out)
        scale = self.get_scale(x.shape[-ndim:], out.shape[-ndim:])
        out = torch.movedim(out, -ndim-1, -1)
        if out.shape[-1] == ndim:
            # Gradient
            for d, out1 in enumerate(out[..., :ndim].unbind(-1)):
                out1 *= scale[d]
        else:
            # if Hessian
            c = ndim
            for d in range(ndim):
                out[..., d] *= scale[d] * scale[d]
                for dd in range(d+1, ndim):
                    out[..., c] *= scale[d] * scale[dd]
                    c += 1
        return out


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

    x = _Restrict.apply(x, factor, shape, ndim, anchor, order, bound, reduce_sum, out)
    return x


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
