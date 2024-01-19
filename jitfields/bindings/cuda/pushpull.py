from ..common.utils import cinfo, cstrides, boundspline_template
from ..common.utils import ctypename
from .utils import (culaunch, get_offset_type, to_cupy, CachedKernel)
import cupy as cp


# ===
# Load module + dispatch 1D/2D/3D
# ===


def get_kernel(key):
    (func, nbatch, ndim, order, bound, extrapolate, *abs,
     reduce_t, scalar_t, offset_t, backward) = key
    template = func
    if backward:
        template += '_backward'
    template += f'<{nbatch}, {ndim}, {extrapolate}, '
    if backward or key == 'grad':
        template += ('true' if abs[0] else 'false') + ', '
    template += ctypename(reduce_t) + ', '
    template += ctypename(scalar_t) + ', '
    template += ctypename(offset_t) + ', '
    template += boundspline_template(bound, order)
    template += '>'
    return template


kernels = CachedKernel('pushpull.cu', get_kernel)


# ===
# Load module + dispatch ND
# ===


def get_kernelnd(key):
    (func, nbatch, ndim, extrapolate, *abs,
     reduce_t, scalar_t, offset_t, backward) = key
    template = func
    if backward:
        template += '_backward'
    template += f'<{nbatch}, {ndim}, {extrapolate}, '
    if backward or key == 'grad':
        template += ('true' if abs[0] else 'false') + ', '
    template += ctypename(reduce_t) + ', '
    template += ctypename(scalar_t) + ', '
    template += ctypename(offset_t) + ', '
    template += '>'
    return template


ndkernels = CachedKernel('pushpull.cu', get_kernelnd)


# ===
# Main functions (format arguments and dispatch)
# ===


def pull(out, inp, grid, order, bound, extrapolate):
    """
    Parameters
    ----------
    out : (*batch, *spatial_out, channels) tensor
    inp : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out : (*batch, *spatial_out, channels) tensor
    """
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    numel = grid.shape[:-1].numel()

    np_inp = to_cupy(inp)
    np_out = to_cupy(out)
    np_grid = to_cupy(grid)

    scalar_t = np_inp.dtype.type
    reduce_t = scalar_t
    offset_t = get_offset_type(np_inp, np_out, np_grid)
    splinc_shape, instride = cinfo(np_inp, dtype=offset_t, backend=cp)
    grid_shape, gridstride = cinfo(np_grid, dtype=offset_t, backend=cp)
    outstride = cstrides(np_out, dtype=offset_t, backend=cp)

    # dispatch
    if ndim <= 3:
        func = kernels.get('pull', nbatch, ndim, tuple(order), tuple(bound),
                           extrapolate, reduce_t, scalar_t, offset_t, False)
        args = (np_out, np_inp, np_grid,
                grid_shape, splinc_shape, outstride, instride, gridstride)
        culaunch(func, numel, args)
    else:
        func = ndkernels.get('pull', nbatch, ndim, extrapolate,
                             reduce_t, scalar_t, offset_t, False)
        order = cp.asarray(order, dtype=cp.uint8)
        bound = cp.asarray(bound, dtype=cp.uint8)
        args = (np_out, np_inp, np_grid, order, bound,
                grid_shape, splinc_shape, outstride, instride, gridstride)
        culaunch(func, numel, args)
    return out


def push(out, inp, grid, order, bound, extrapolate):
    """
    Parameters
    ----------
    out : (*batch, *spatial_out, channels) tensor
    inp : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_in, ndim) tensor
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out : (*batch, *spatial_out, channels) tensor
    """
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    numel = grid.shape[:-1].numel()

    np_inp = to_cupy(inp)
    np_out = to_cupy(out)
    np_grid = to_cupy(grid)

    scalar_t = np_inp.dtype.type
    reduce_t = scalar_t
    offset_t = get_offset_type(np_inp, np_out, np_grid)
    splinc_shape, outstride = cinfo(np_out, dtype=offset_t, backend=cp)
    _, instride = cinfo(np_inp, dtype=offset_t, backend=cp)
    grid_shape, gridstride = cinfo(np_grid, dtype=offset_t, backend=cp)

    # dispatch
    if ndim <= 3:
        func = kernels.get('push', nbatch, ndim, tuple(order), tuple(bound),
                           extrapolate, reduce_t, scalar_t, offset_t, False)
        args = (np_out, np_inp, np_grid,
                grid_shape, splinc_shape, outstride, instride, gridstride)
        culaunch(func, numel, args)
    else:
        func = ndkernels.get('push', nbatch, ndim, extrapolate,
                             reduce_t, scalar_t, offset_t, False)
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        args = (np_out, np_inp, np_grid, order, bound,
                grid_shape, splinc_shape, outstride, instride, gridstride)
        culaunch(func, numel, args)

    return out


def count(out, grid, order, bound, extrapolate):
    """
    Parameters
    ----------
    out : (*batch, *spatial_out, 1) tensor
    grid : (*batch, *spatial_in, ndim) tensor
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out : (*batch, *spatial_out, 1) tensor
    """
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    numel = grid.shape[:-1].numel()

    np_out = to_cupy(out)
    np_grid = to_cupy(grid)

    scalar_t = np_grid.dtype.type
    reduce_t = scalar_t
    offset_t = get_offset_type(np_out, np_grid)
    splinc_shape, outstride = cinfo(np_out, dtype=offset_t, backend=cp)
    grid_shape, gridstride = cinfo(np_grid, dtype=offset_t, backend=cp)

    # dispatch
    if ndim <= 3:
        func = kernels.get('count', nbatch, ndim, tuple(order), tuple(bound),
                           extrapolate, reduce_t, scalar_t, offset_t, False)
        args = (np_out, np_grid,
                grid_shape, splinc_shape, outstride, gridstride)
        culaunch(func, numel, args)
    else:
        func = ndkernels.get('count', nbatch, ndim, extrapolate,
                             reduce_t, scalar_t, offset_t, False)
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        args = (np_out, np_grid, order, bound,
                grid_shape, splinc_shape, outstride, gridstride)
        culaunch(func, numel, args)

    return out


def grad(out, inp, grid, order, bound, extrapolate, abs=False):
    """
    Parameters
    ----------
    out : (*batch, *spatial_out, channels, ndim) tensor
    inp : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}
    abs : bool

    Returns
    -------
    out : (*batch, *spatial_out, channels, ndim) tensor
    """
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    numel = grid.shape[:-1].numel()

    np_inp = to_cupy(inp)
    np_out = to_cupy(out)
    np_grid = to_cupy(grid)

    scalar_t = np_grid.dtype.type
    reduce_t = scalar_t
    offset_t = get_offset_type(np_inp, np_out, np_grid)
    splinc_shape, instride = cinfo(np_inp, dtype=offset_t, backend=cp)
    grid_shape, gridstride = cinfo(np_grid, dtype=offset_t, backend=cp)
    _, outstride = cinfo(np_out, dtype=offset_t, backend=cp)

    # dispatch
    if ndim <= 3:
        func = kernels.get('grad', nbatch, ndim, tuple(order), tuple(bound),
                           extrapolate, abs, reduce_t, scalar_t, offset_t,
                           False)
        args = (np_out, np_inp, np_grid,
                grid_shape, splinc_shape, outstride, instride, gridstride)
        culaunch(func, numel, args)
    else:
        func = ndkernels.get('grad', nbatch, ndim, extrapolate, abs,
                             reduce_t, scalar_t, offset_t, False)
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        args = (np_out, np_inp, np_grid, order, bound,
                grid_shape, splinc_shape, outstride, instride, gridstride)
        culaunch(func, numel, args)

    return out


def hess(out, inp, grid, order, bound, extrapolate, abs=False):
    """
    Parameters
    ----------
    out : (*batch, *spatial_out, channels, ndim(*ndim+1)//2) tensor
    inp : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}
    abs : bool

    Returns
    -------
    out : (*batch, *spatial_out, channels, ndim(*ndim+1)//2) tensor
    """
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    numel = grid.shape[:-1].numel()

    np_inp = to_cupy(inp)
    np_out = to_cupy(out)
    np_grid = to_cupy(grid)

    scalar_t = np_grid.dtype.type
    reduce_t = scalar_t
    offset_t = get_offset_type(np_inp, np_out, np_grid)
    splinc_shape, instride = cinfo(np_inp, dtype=offset_t, backend=cp)
    grid_shape, gridstride = cinfo(np_grid, dtype=offset_t, backend=cp)
    _, outstride = cinfo(np_out, dtype=offset_t, backend=cp)

    # dispatch
    if ndim <= 3:
        func = kernels.get('hess', nbatch, ndim, tuple(order), tuple(bound),
                           extrapolate, abs, reduce_t, scalar_t, offset_t,
                           False)
        args = (np_out, np_inp, np_grid,
                grid_shape, splinc_shape, outstride, instride, gridstride)
        culaunch(func, numel, args)
    else:
        func = ndkernels.get('hess', nbatch, ndim, extrapolate, abs,
                             reduce_t, scalar_t, offset_t, False)
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        args = (np_out, np_inp, np_grid, order, bound,
                grid_shape, splinc_shape, outstride, instride, gridstride)
        culaunch(func, numel, args)

    return out


def pull_backward(out_grad_inp, out_grad_grid, inp_grad, inp, grid,
                  order, bound, extrapolate, abs=False):
    """
    Parameters
    ----------
    out_grad_inp : (*batch, *spatial_in, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    inp_grad : (*batch, *spatial_in, channels) tensor
    inp : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}
    abs : bool

    Returns
    -------
    out_grad_inp : (*batch, *spatial_in, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    numel = grid.shape[:-1].numel()

    np_inp = to_cupy(inp)
    np_grid = to_cupy(grid)
    np_inp_grad = to_cupy(inp_grad)
    np_out_grad_inp = to_cupy(out_grad_inp)
    np_out_grad_grid = to_cupy(out_grad_grid)

    scalar_t = np_grid.dtype.type
    reduce_t = scalar_t
    offset_t = get_offset_type(
        np_inp, np_inp_grad, np_grid, np_out_grad_inp, np_out_grad_grid)

    splinc_shape, inp_stride = cinfo(np_inp, dtype=offset_t, backend=cp)
    grid_shape, grid_stride = cinfo(np_grid, dtype=offset_t, backend=cp)
    _, inp_grad_stride = cinfo(np_inp_grad, dtype=offset_t, backend=cp)
    _, out_grad_inp_stride = cinfo(np_out_grad_inp, dtype=offset_t, backend=cp)
    _, out_grad_grid_stride = cinfo(
        np_out_grad_grid, dtype=offset_t, backend=cp)

    # dispatch
    if ndim <= 3:
        func = kernels.get('pull', nbatch, ndim, tuple(order), tuple(bound),
                           extrapolate, abs, reduce_t, scalar_t, offset_t,
                           True)
        args = (
            np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
            grid_shape, splinc_shape,
            out_grad_inp_stride, out_grad_grid_stride,
            inp_stride, inp_grad_stride, grid_stride)
        culaunch(func, numel, args)
    else:
        func = ndkernels.get('pull', nbatch, ndim, extrapolate, abs,
                             reduce_t, scalar_t, offset_t, True)
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        args = (
            np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
            order, bound, grid_shape, splinc_shape,
            out_grad_inp_stride, out_grad_grid_stride,
            inp_stride, inp_grad_stride, grid_stride)
        culaunch(func, numel, args)

    return out_grad_inp, out_grad_grid


def push_backward(out_grad_inp, out_grad_grid, inp_grad, inp, grid,
                  order, bound, extrapolate, abs=False):
    """
    Parameters
    ----------
    out_grad_inp : (*batch, *spatial_out, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    inp_grad : (*batch, *spatial_in, channels) tensor
    inp : (*batch, *spatial_out, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}
    abs : bool

    Returns
    -------
    out_grad_inp : (*batch, *spatial_out, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    numel = grid.shape[:-1].numel()

    np_inp = to_cupy(inp)
    np_grid = to_cupy(grid)
    np_inp_grad = to_cupy(inp_grad)
    np_out_grad_inp = to_cupy(out_grad_inp)
    np_out_grad_grid = to_cupy(out_grad_grid)

    scalar_t = np_grid.dtype.type
    reduce_t = scalar_t
    offset_t = get_offset_type(
        np_inp, np_grid, np_inp_grad, np_out_grad_inp, np_out_grad_grid)
    splinc_shape, inp_grad_stride = cinfo(
        np_inp_grad, dtype=offset_t, backend=cp)
    grid_shape, grid_stride = cinfo(np_grid, dtype=offset_t, backend=cp)
    _, inp_stride = cinfo(np_inp, dtype=offset_t, backend=cp)
    _, out_grad_inp_stride = cinfo(
        np_out_grad_inp, dtype=offset_t, backend=cp)
    _, out_grad_grid_stride = cinfo(
        np_out_grad_grid, dtype=offset_t, backend=cp)

    # dispatch
    if ndim <= 3:
        func = kernels.get('push', nbatch, ndim, tuple(order), tuple(bound),
                           extrapolate, abs, reduce_t, scalar_t, offset_t,
                           True)
        args = (
            np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
            grid_shape, splinc_shape,
            out_grad_inp_stride, out_grad_grid_stride,
            inp_stride, inp_grad_stride, grid_stride)
        culaunch(func, numel, args)
    else:
        func = ndkernels.get('push', nbatch, ndim, extrapolate, abs,
                             reduce_t, scalar_t, offset_t, True)
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        args = (
            np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
            order, bound, grid_shape, splinc_shape,
            out_grad_inp_stride, out_grad_grid_stride,
            inp_stride, inp_grad_stride, grid_stride)
        culaunch(func, numel, args)

    return out_grad_inp, out_grad_grid


def count_backward(out_grad_grid, inp_grad, grid,
                   order, bound, extrapolate, abs=False):
    """
    Parameters
    ----------
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    inp_grad : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}
    abs : bool

    Returns
    -------
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    numel = grid.shape[:-1].numel()

    np_grid = to_cupy(grid)
    np_inp_grad = to_cupy(inp_grad)
    np_out_grad_grid = to_cupy(out_grad_grid)

    scalar_t = np_grid.dtype.type
    reduce_t = scalar_t
    offset_t = get_offset_type(np_inp_grad, np_out_grad_grid, np_grid)
    splinc_shape, inp_grad_stride = cinfo(
        np_inp_grad, dtype=offset_t, backend=cp)
    grid_shape, grid_stride = cinfo(np_grid, dtype=offset_t, backend=cp)
    _, out_grad_grid_stride = cinfo(
        np_out_grad_grid, dtype=offset_t, backend=cp)

    # dispatch
    if ndim <= 3:
        func = kernels.get('count', nbatch, ndim, tuple(order), tuple(bound),
                           extrapolate, abs, reduce_t, scalar_t, offset_t,
                           True)
        args = (np_out_grad_grid, np_inp_grad, np_grid,
                grid_shape, splinc_shape,
                out_grad_grid_stride, inp_grad_stride, grid_stride)
        culaunch(func, numel, args)
    else:
        func = ndkernels.get('count', nbatch, ndim, extrapolate, abs,
                             reduce_t, scalar_t, offset_t, True)
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        args = (np_out_grad_grid, np_inp_grad, np_grid,
                order, bound, grid_shape, splinc_shape,
                out_grad_grid_stride, inp_grad_stride, grid_stride)
        culaunch(func, numel, args)

    return out_grad_grid


def grad_backward(out_grad_inp, out_grad_grid, inp_grad, inp, grid,
                  order, bound, extrapolate, abs=False):
    """
    Parameters
    ----------
    out_grad_inp : (*batch, *spatial_in, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    inp_grad : (*batch, *spatial_in, channels, ndim) tensor
    inp : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}
    abs : bool

    Returns
    -------
    out_grad_inp : (*batch, *spatial_in, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    numel = grid.shape[:-1].numel()

    np_inp = to_cupy(inp)
    np_grid = to_cupy(grid)
    np_inp_grad = to_cupy(inp_grad)
    np_out_grad_inp = to_cupy(out_grad_inp)
    np_out_grad_grid = to_cupy(out_grad_grid)

    scalar_t = np_grid.dtype.type
    reduce_t = scalar_t
    offset_t = get_offset_type(
        np_inp, np_grid, np_inp_grad, np_out_grad_inp, np_out_grad_inp)
    splinc_shape, inp_stride = cinfo(np_inp, dtype=offset_t, backend=cp)
    grid_shape, grid_stride = cinfo(np_grid, dtype=offset_t, backend=cp)
    _, inp_grad_stride = cinfo(np_inp_grad, dtype=offset_t, backend=cp)
    _, out_grad_inp_stride = cinfo(
        np_out_grad_inp, dtype=offset_t, backend=cp)
    _, out_grad_grid_stride = cinfo(
        np_out_grad_grid, dtype=offset_t, backend=cp)

    # dispatch
    if ndim <= 3:
        func = kernels.get('grad', nbatch, ndim, tuple(order), tuple(bound),
                           extrapolate, abs, reduce_t, scalar_t, offset_t,
                           True)
        args = (
            np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
            grid_shape, splinc_shape,
            out_grad_inp_stride, out_grad_grid_stride,
            inp_stride, inp_grad_stride, grid_stride)
        culaunch(func, numel, args)
    else:
        func = ndkernels.get('grad', nbatch, ndim, extrapolate, abs,
                             reduce_t, scalar_t, offset_t, True)
        order = cp.asarray(order, dtype='uint8')
        bound = cp.asarray(bound, dtype='uint8')
        args = (
            np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
            order, bound, grid_shape, splinc_shape,
            out_grad_inp_stride, out_grad_grid_stride,
            inp_stride, inp_grad_stride, grid_stride)
        culaunch(func, numel, args)

    return out_grad_inp, out_grad_grid
