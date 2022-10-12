from .utils import boundspline_template, cwrap, ctypename
from ..common.utils import cinfo
import cppyy
import numpy as np
import os

this_folder = os.path.abspath(os.path.dirname(__file__))
cppyy.add_include_path(os.path.join(this_folder, '..', 'csrc'))
cppyy.include('pushpull.hpp')


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

    np_inp = inp.numpy()
    np_out = out.numpy()
    np_grid = grid.numpy()

    offset_t = np.int64
    splinc_shape, instride = cinfo(np_inp, dtype=offset_t)
    grid_shape, gridstride = cinfo(np_grid, dtype=offset_t)
    _, outstride = cinfo(np_out, dtype=offset_t)
    nalldim = int(np_grid.ndim)

    # dispatch
    if ndim <= 3:
        template = boundspline_template(bound, order)
        template += f', {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(getattr(cppyy.gbl.jf.pushpull, f'pull{ndim}d')[template])
        func(np_out, np_inp, np_grid, nalldim,
             grid_shape, splinc_shape, outstride, instride, gridstride)
    else:
        template = f'{int(ndim)}, {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(cppyy.gbl.jf.pushpull.pullnd[template])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(np_out, np_inp, np_grid, nalldim, order, bound,
             grid_shape, splinc_shape, outstride, instride, gridstride)

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

    np_inp = inp.numpy()
    np_out = out.numpy()
    np_grid = grid.numpy()

    offset_t = np.int64
    splinc_shape, outstride = cinfo(np_out, dtype=offset_t)
    _, instride = cinfo(np_inp, dtype=offset_t)
    grid_shape, gridstride = cinfo(np_grid, dtype=offset_t)
    nalldim = int(np_grid.ndim)

    # dispatch
    if ndim <= 3:
        template = boundspline_template(bound, order)
        template += f', {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(getattr(cppyy.gbl.jf.pushpull, f'push{ndim}d')[template])
        func(np_out, np_inp, np_grid, nalldim,
             grid_shape, splinc_shape, outstride, instride, gridstride)
    else:
        template = f'{int(ndim)}, {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(cppyy.gbl.jf.pushpull.pushnd[template])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(np_out, np_inp, np_grid, nalldim, order, bound,
             grid_shape, splinc_shape, outstride, instride, gridstride)

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

    np_out = out.numpy()
    np_grid = grid.numpy()

    offset_t = np.int64
    splinc_shape, outstride = cinfo(np_out, dtype=offset_t)
    grid_shape, gridstride = cinfo(np_grid, dtype=offset_t)
    nalldim = int(np_grid.ndim)

    # dispatch
    if ndim <= 3:
        template = boundspline_template(bound, order)
        template += f', {int(extrapolate)}'
        template += ', ' + ctypename(np_out.dtype)
        func = cwrap(getattr(cppyy.gbl.jf.pushpull, f'count{ndim}d')[template])
        func(np_out, np_grid, nalldim,
             grid_shape, splinc_shape, outstride, gridstride)
    else:
        template = f'{int(ndim)}, {int(extrapolate)}'
        template += ', ' + ctypename(np_out.dtype)
        func = cwrap(cppyy.gbl.jf.pushpull.countnd[template])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(np_out, np_grid, nalldim, order, bound,
             grid_shape, splinc_shape, outstride, gridstride)

    return out


def grad(out, inp, grid, order, bound, extrapolate):
    """
    Parameters
    ----------
    out : (*batch, *spatial_out, channels, ndim) tensor
    inp : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out : (*batch, *spatial_out, channels, ndim) tensor
    """
    ndim = grid.shape[-1]

    np_inp = inp.numpy()
    np_out = out.numpy()
    np_grid = grid.numpy()

    offset_t = np.int64
    splinc_shape, instride = cinfo(np_inp, dtype=offset_t)
    grid_shape, gridstride = cinfo(np_grid, dtype=offset_t)
    _, outstride = cinfo(np_out, dtype=offset_t)
    nalldim = int(np_grid.ndim)

    # dispatch
    if ndim <= 3:
        template = boundspline_template(bound, order)
        template += f', {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(getattr(cppyy.gbl.jf.pushpull, f'grad{ndim}d')[template])
        func(np_out, np_inp, np_grid, nalldim,
             grid_shape, splinc_shape, outstride, instride, gridstride)
    else:
        template = f'{int(ndim)}, {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(cppyy.gbl.jf.pushpull.gradnd[template])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(np_out, np_inp, np_grid, nalldim, order, bound,
             grid_shape, splinc_shape, outstride, instride, gridstride)

    return out


def pull_backward(out_grad_inp, out_grad_grid, inp_grad, inp, grid,
                  order, bound, extrapolate):
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

    Returns
    -------
    out_grad_inp : (*batch, *spatial_in, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ndim = grid.shape[-1]

    np_inp = inp.numpy()
    np_grid = grid.numpy()
    np_inp_grad = inp_grad.numpy()
    np_out_grad_inp = out_grad_inp.numpy()
    np_out_grad_grid = out_grad_grid.numpy()

    offset_t = np.int64
    splinc_shape, inp_stride = cinfo(np_inp, dtype=offset_t)
    grid_shape, grid_stride = cinfo(np_grid, dtype=offset_t)
    _, inp_grad_stride = cinfo(np_inp_grad, dtype=offset_t)
    _, out_grad_inp_stride = cinfo(np_out_grad_inp, dtype=offset_t)
    _, out_grad_grid_stride = cinfo(np_out_grad_grid, dtype=offset_t)
    nalldim = int(np_grid.ndim)

    # dispatch
    if ndim <= 3:
        template = boundspline_template(bound, order)
        template += f', {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(getattr(cppyy.gbl.jf.pushpull, f'pull{ndim}d_backward')[template])
        func(np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
             nalldim, grid_shape, splinc_shape,
             out_grad_inp_stride, out_grad_grid_stride,
             inp_stride, inp_grad_stride, grid_stride)
    else:
        template = f'{int(ndim)}, {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(cppyy.gbl.jf.pushpull.pullnd_backward[template])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
             nalldim, order, bound, grid_shape, splinc_shape,
             out_grad_inp_stride, out_grad_grid_stride,
             inp_stride, inp_grad_stride, grid_stride)

    return out_grad_inp, out_grad_grid


def push_backward(out_grad_inp, out_grad_grid, inp_grad, inp, grid,
                  order, bound, extrapolate):
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

    Returns
    -------
    out_grad_inp : (*batch, *spatial_out, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ndim = grid.shape[-1]

    np_inp = inp.numpy()
    np_grid = grid.numpy()
    np_inp_grad = inp_grad.numpy()
    np_out_grad_inp = out_grad_inp.numpy()
    np_out_grad_grid = out_grad_grid.numpy()

    offset_t = np.int64
    splinc_shape, inp_grad_stride = cinfo(np_inp_grad, dtype=offset_t)
    grid_shape, grid_stride = cinfo(np_grid, dtype=offset_t)
    _, inp_stride = cinfo(np_inp, dtype=offset_t)
    _, out_grad_inp_stride = cinfo(np_out_grad_inp, dtype=offset_t)
    _, out_grad_grid_stride = cinfo(np_out_grad_grid, dtype=offset_t)
    nalldim = int(np_grid.ndim)

    # dispatch
    if ndim <= 3:
        template = boundspline_template(bound, order)
        template += f', {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(getattr(cppyy.gbl.jf.pushpull, f'push{ndim}d_backward')[template])
        func(np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
             nalldim, grid_shape, splinc_shape,
             out_grad_inp_stride, out_grad_grid_stride,
             inp_stride, inp_grad_stride, grid_stride)
    else:
        template = f'{int(ndim)}, {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(cppyy.gbl.jf.pushpull.pushnd_backward[template])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
             nalldim, order, bound, grid_shape, splinc_shape,
             out_grad_inp_stride, out_grad_grid_stride,
             inp_stride, inp_grad_stride, grid_stride)

    return out_grad_inp, out_grad_grid


def count_backward(out_grad_grid, inp_grad, grid,
                   order, bound, extrapolate):
    """
    Parameters
    ----------
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    inp_grad : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ndim = grid.shape[-1]

    np_grid = grid.numpy()
    np_inp_grad = inp_grad.numpy()
    np_out_grad_grid = out_grad_grid.numpy()

    offset_t = np.int64
    splinc_shape, inp_grad_stride = cinfo(np_inp_grad, dtype=offset_t)
    grid_shape, grid_stride = cinfo(np_grid, dtype=offset_t)
    _, out_grad_grid_stride = cinfo(np_out_grad_grid, dtype=offset_t)
    nalldim = int(np_grid.ndim)

    # dispatch
    if ndim <= 3:
        template = boundspline_template(bound, order)
        template += f', {int(extrapolate)}'
        template += ', ' + ctypename(np_grid.dtype)
        func = cwrap(getattr(cppyy.gbl.jf.pushpull, f'count{ndim}d_backward')[template])
        func(np_out_grad_grid, np_inp_grad, np_grid,
             nalldim, grid_shape, splinc_shape,
             out_grad_grid_stride, inp_grad_stride, grid_stride)
    else:
        template = f'{int(ndim)}, {int(extrapolate)}'
        template += ', ' + ctypename(np_grid.dtype)
        func = cwrap(cppyy.gbl.jf.pushpull.countnd_backward[template])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(np_out_grad_grid, np_inp_grad, np_grid,
             nalldim, order, bound, grid_shape, splinc_shape,
             out_grad_grid_stride, inp_grad_stride, grid_stride)

    return out_grad_grid


def grad_backward(out_grad_inp, out_grad_grid, inp_grad, inp, grid,
                  order, bound, extrapolate):
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

    Returns
    -------
    out_grad_inp : (*batch, *spatial_in, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ndim = grid.shape[-1]

    np_inp = inp.numpy()
    np_grid = grid.numpy()
    np_inp_grad = inp_grad.numpy()
    np_out_grad_inp = out_grad_inp.numpy()
    np_out_grad_grid = out_grad_grid.numpy()

    offset_t = np.int64
    splinc_shape, inp_stride = cinfo(np_inp, dtype=offset_t)
    grid_shape, grid_stride = cinfo(np_grid, dtype=offset_t)
    _, inp_grad_stride = cinfo(np_inp_grad, dtype=offset_t)
    _, out_grad_inp_stride = cinfo(np_out_grad_inp, dtype=offset_t)
    _, out_grad_grid_stride = cinfo(np_out_grad_grid, dtype=offset_t)
    nalldim = int(np_grid.ndim)

    # dispatch
    if ndim <= 3:
        template = boundspline_template(bound, order)
        template += f', {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(getattr(cppyy.gbl.jf.pushpull, f'grad{ndim}d_backward')[template])
        func(np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
             nalldim, grid_shape, splinc_shape,
             out_grad_inp_stride, out_grad_grid_stride,
             inp_stride, inp_grad_stride, grid_stride)
    else:
        template = f'{int(ndim)}, {int(extrapolate)}'
        template += ', ' + ctypename(np_inp.dtype)
        func = cwrap(cppyy.gbl.jf.pushpull.gradnd_backward[template])
        order = np.asarray(order, dtype='uint8')
        bound = np.asarray(bound, dtype='uint8')
        func(np_out_grad_inp, np_out_grad_grid, np_inp, np_inp_grad, np_grid,
             nalldim, order, bound, grid_shape, splinc_shape,
             out_grad_inp_stride, out_grad_grid_stride,
             inp_stride, inp_grad_stride, grid_stride)

    return out_grad_inp, out_grad_grid
