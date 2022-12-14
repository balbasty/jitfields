import torch
from .utils import try_import, ensure_list
from .splinc import spline_coeff_nd, spline_coeff_nd_
from .common.bounds import convert_bound
from .common.spline import convert_order
cuda_pushpull = try_import('jitfields.cuda', 'pushpull')
cpu_pushpull = try_import('jitfields.cpp', 'pushpull')


def pull(inp, grid, order=2, bound='dct2', extrapolate=True, prefilter=False,
         out=None):
    """Sample a tensor using spline interpolation

    Parameters
    ----------
    inp : (..., *inshape, channel) tensor
        Input tensor
    grid : (..., *outshape, ndim) tensor
        Tensor of coordinates into `inp`
    order : [sequence of] {0..7}, default=2
        Interpolation order.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
        How to deal with out-of-bound values.
    extrapolate : bool or {'center', 'edge'}
        - True: use bound to extrapolate out-of-bound value
        - False or 'center': do not extrapolate values that fall outside
          of the centers of the first and last voxels.
        - 'edge': do not extrapolate values that fall outside
           of the edges of the first and last voxels.
    prefilter : bool, default=True
        Whether to first compute interpolating coefficients.
        Must be true for proper interpolation, otherwise this
        function merely performs a non-interpolating "spline sampling".

    Returns
    -------
    out : (..., *outshape, channel) tensor
        Pulled tensor

    """
    ndim = grid.shape[-1]
    if ndim > 3:
        raise NotImplementedError("Not implemented for spatial dim > 3")
    if prefilter:
        inp = spline_coeff_nd(inp.movedim(-1, 0), order, bound, ndim).movedim(0, -1)
    inp, grid = _broadcast_pull(inp, grid)
    order, bound, extrapolate = _preproc_opt(order, bound, extrapolate, ndim)
    return Pull.apply(inp, grid, order, bound, extrapolate, out)


def push(inp, grid, shape=None, order=2, bound='dct2', extrapolate=True,
         prefilter=False, out=None):
    """Splat a tensor using spline interpolation

    Parameters
    ----------
    inp : (..., *inshape, channel) tensor
        Input tensor
    grid : (..., *inshape, ndim) tensor
        Tensor of coordinates into `inp`
    shape : sequence[int], default=inshape
        Output spatial shape
    order : [sequence of] {0..7}, default=2
        Interpolation order.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
        How to deal with out-of-bound values.
    extrapolate : bool or {'center', 'edge'}
        - True: use bound to extrapolate out-of-bound value
        - False or 'center': do not extrapolate values that fall outside
          of the centers of the first and last voxels.
        - 'edge': do not extrapolate values that fall outside
           of the edges of the first and last voxels.
    prefilter : bool, default=True
        Whether to compute interpolating coefficients at the end.

    Returns
    -------
    out : (..., *shape, channel) tensor
        Pulled tensor

    """
    ndim = grid.shape[-1]
    if ndim > 3:
        raise NotImplementedError("Not implemented for spatial dim > 3")
    inp, grid = _broadcast_push(inp, grid)
    shape = list(shape or inp.shape[-ndim-1:-1])
    order, bound, extrapolate = _preproc_opt(order, bound, extrapolate, ndim)
    inp = Push.apply(inp, grid, shape, order, bound, extrapolate, out)
    if prefilter:
        inp = spline_coeff_nd_(inp.movedim(-1, 0), order, bound, ndim).movedim(0, -1)
    return inp


def count(grid, shape=None, order=2, bound='dct2', extrapolate=True,
          out=None):
    """Splat ones using spline interpolation

    Parameters
    ----------
    grid : (..., *inshape, ndim) tensor
        Tensor of coordinates
    shape : sequence[int], default=inshape
        Output spatial shape
    order : [sequence of] {0..7}, default=2
        Interpolation order.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
        How to deal with out-of-bound values.
    extrapolate : bool or {'center', 'edge'}
        - True: use bound to extrapolate out-of-bound value
        - False or 'center': do not extrapolate values that fall outside
          of the centers of the first and last voxels.
        - 'edge': do not extrapolate values that fall outside
           of the edges of the first and last voxels.

    Returns
    -------
    out : (..., *shape) tensor
        Pulled tensor

    """
    ndim = grid.shape[-1]
    if ndim > 3:
        raise NotImplementedError("Not implemented for spatial dim > 3")
    shape = list(shape or grid.shape[-ndim-1:-1])
    order, bound, extrapolate = _preproc_opt(order, bound, extrapolate, ndim)
    return Count.apply(grid, shape, order, bound, extrapolate, out)


def grad(inp, grid, order=2, bound='dct2', extrapolate=True, prefilter=False,
         out=None):
    """Sample the spatial gradients of a tensor using spline interpolation

    Parameters
    ----------
    inp : (..., *inshape, channel) tensor
        Input tensor
    grid : (..., *outshape, ndim) tensor
        Tensor of coordinates into `inp`
    order : [sequence of] {0..7}, default=2
        Interpolation order.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
        How to deal with out-of-bound values.
    extrapolate : bool or {'center', 'edge'}
        - True: use bound to extrapolate out-of-bound value
        - False or 'center': do not extrapolate values that fall outside
          of the centers of the first and last voxels.
        - 'edge': do not extrapolate values that fall outside
           of the edges of the first and last voxels.
    prefilter : bool, default=True
        Whether to first compute interpolating coefficients.
        Must be true for proper interpolation, otherwise this
        function merely performs a non-interpolating "spline sampling".

    Returns
    -------
    out : (..., *outshape, channel, ndim) tensor
        Pulled gradients

    """
    ndim = grid.shape[-1]
    if ndim > 3:
        raise NotImplementedError("Not implemented for spatial dim > 3")
    if prefilter:
        inp = spline_coeff_nd(inp.movedim(-1, 0), order, bound, ndim).movedim(0, -1)
    inp, grid = _broadcast_pull(inp, grid)
    order, bound, extrapolate = _preproc_opt(order, bound, extrapolate, ndim)
    return Grad.apply(inp, grid, order, bound, extrapolate, out)


class Pull(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, grid, order, bound, extrapolate, out):
        _fwd = cuda_pushpull.pull if inp.is_cuda else cpu_pushpull.pull
        ctx.opt = (order, bound, extrapolate)
        ctx.save_for_backward(inp, grid)
        fullshape = grid.shape[:-1] + inp.shape[-1:]
        out = inp.new_empty(fullshape) if out is None else out.view(fullshape)
        out = _fwd(out, inp, grid, order, bound, extrapolate)
        return out

    @staticmethod
    def backward(ctx, grad):
        _bwd = (cuda_pushpull.pull_backward if grad.is_cuda else
                cpu_pushpull.pull_backward)
        inp, grid = ctx.saved_tensors
        outgrad_inp = torch.zeros_like(inp)
        outgrad_grid = torch.empty_like(grid)
        _bwd(outgrad_inp, outgrad_grid, grad, inp, grid, *ctx.opt)
        return (outgrad_inp, outgrad_grid) + (None,) * 4


class Push(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, grid, shape, order, bound, extrapolate, out):
        _fwd = cuda_pushpull.push if inp.is_cuda else cpu_pushpull.push
        ctx.opt = (order, bound, extrapolate)
        ctx.save_for_backward(inp, grid)
        ndim = grid.shape[-1]
        fullshape = list(grid.shape[:-ndim-1]) + list(shape) + list(inp.shape[-1:])
        out = inp.new_zeros(fullshape) if out is None else out.view(fullshape)
        out = _fwd(out, inp, grid, order, bound, extrapolate)
        return out

    @staticmethod
    def backward(ctx, grad):
        _bwd = (cuda_pushpull.push_backward if grad.is_cuda else
                cpu_pushpull.push_backward)
        inp, grid = ctx.saved_tensors
        outgrad_inp = torch.empty_like(inp)
        outgrad_grid = torch.empty_like(grid)
        _bwd(outgrad_inp, outgrad_grid, grad, inp, grid, *ctx.opt)
        return (outgrad_inp, outgrad_grid) + (None,) * 5


class Count(torch.autograd.Function):

    @staticmethod
    def forward(ctx, grid, shape, order, bound, extrapolate, out):
        _fwd = cuda_pushpull.count if grid.is_cuda else cpu_pushpull.count
        ctx.opt = (order, bound, extrapolate)
        ctx.save_for_backward(grid)
        ndim = grid.shape[-1]
        fullshape = list(grid.shape[:-ndim-1]) + list(shape) + [1]
        out = grid.new_zeros(fullshape) if out is None else out.view(fullshape)
        out = _fwd(out, grid, order, bound, extrapolate).squeeze(-1)
        return out

    @staticmethod
    def backward(ctx, grad):
        _bwd = (cuda_pushpull.count_backward if grad.is_cuda else
                cpu_pushpull.count_backward)
        grid, = ctx.saved_tensors
        outgrad_grid = torch.empty_like(grid)
        _bwd(outgrad_grid, grad.unsqueeze(-1), grid, *ctx.opt)
        return (outgrad_grid,) + (None,) * 5


class Grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, grid, order, bound, extrapolate, out):
        _fwd = cuda_pushpull.grad if inp.is_cuda else cpu_pushpull.grad
        ctx.opt = (order, bound, extrapolate)
        ctx.save_for_backward(inp, grid)
        fullshape = grid.shape[:-1] + inp.shape[-1:] + grid.shape[-1:]
        out = inp.new_empty(fullshape) if out is None else out.view(fullshape)
        out = _fwd(out, inp, grid, order, bound, extrapolate)
        return out

    @staticmethod
    def backward(ctx, grad):
        _bwd = (cuda_pushpull.grad_backward if grad.is_cuda else
                cpu_pushpull.grad_backward)
        inp, grid = ctx.saved_tensors
        outgrad_inp = torch.zeros_like(inp)
        outgrad_grid = torch.empty_like(grid)
        _bwd(outgrad_inp, outgrad_grid, grad, inp, grid, *ctx.opt)
        return (outgrad_inp, outgrad_grid) + (None,) * 4


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
