from ..common.utils import cinfo
from .utils import (get_offset_type, to_cupy, culaunch, CachedKernel)
import cupy as cp
import math

kernels = CachedKernel('posdef.cu')


def _sym_matvec_base(func, out, inp, mat, dtype=None):
    """Common worker for matvec / addmatvec_ / submatvec_

    Parameters
    ----------
    func : cppyy binding
    out : (*batch, C) tensor
    inp : (*batch, C) tensor
    mat : (*batch, C*(C+1)//2) tensor

    Returns
    -------
    out : (*batch, C) tensor
    """
    nc = out.shape[-1]
    nbatch = out.ndim - 1
    numel = out.shape[:-1].numel()
    np_inp = to_cupy(inp)
    np_out = to_cupy(out)
    np_mat = to_cupy(mat)

    scalar_t = np_inp.dtype
    reduce_t = cp.dtype(dtype or cp.float64)
    offset_t = get_offset_type(np_inp, np_out, np_mat)

    shape, out_stride = cinfo(np_out, dtype=offset_t, backend=cp)
    _, inp_stride = cinfo(np_inp, dtype=offset_t, backend=cp)
    _, mat_stride = cinfo(np_mat, dtype=offset_t, backend=cp)

    # dispatch
    func = kernels.get(func, nbatch, nc, reduce_t, scalar_t, offset_t)
    args = (np_out, np_mat, np_inp, shape, out_stride, mat_stride, inp_stride)
    culaunch(func, numel, args)
    return out


def sym_matvec(out, inp, mat, dtype=None):
    """
    Parameters
    ----------
    out : (*batch, C) tensor
    inp : (*batch, C) tensor
    mat : (*batch, C*(C+1)//2) tensor

    Returns
    -------
    out : (*batch, C) tensor
    """
    return _sym_matvec_base('sym_matvec', out, inp, mat, dtype)


def sym_addmatvec_(out, inp, mat, dtype=None):
    """
    Parameters
    ----------
    out : (*batch, C) tensor
    inp : (*batch, C) tensor
    mat : (*batch, C*(C+1)//2) tensor

    Returns
    -------
    out : (*batch, C) tensor
    """
    return _sym_matvec_base('sym_addmatvec_', out, inp, mat, dtype)


def sym_submatvec_(out, inp, mat, dtype=None):
    """
    Parameters
    ----------
    out : (*batch, C) tensor
    inp : (*batch, C) tensor
    mat : (*batch, C*(C+1)//2) tensor

    Returns
    -------
    out : (*batch, C) tensor
    """
    return _sym_matvec_base('sym_submatvec_', out, inp, mat, dtype)


def sym_matvec_backward(out, grd, inp, dtype=None):
    """Backward pass of matvec wrt the matrix

    Parameters
    ----------
    out : (*batch, C*(C+1)//2)) tensor
    grd : (*batch, C) tensor
    inp : (*batch, C) tensor

    Returns
    -------
    out : (*batch, C*(C+1)//2)) tensor
    """
    nc = grd.shape[-1]
    nbatch = out.ndim - 1
    numel = out.shape[:-1].numel()
    np_inp = to_cupy(inp)
    np_out = to_cupy(out)
    np_grd = to_cupy(grd)

    offset_t = cp.int64
    scalar_t = np_inp.dtype
    reduce_t = cp.dtype(dtype or cp.float64)

    shape, grd_stride = cinfo(np_grd, dtype=offset_t, backend=cp)
    _, inp_stride = cinfo(np_inp, dtype=offset_t, backend=cp)
    _, out_stride = cinfo(np_out, dtype=offset_t, backend=cp)

    # dispatch
    func = kernels.get('sym_matvec_backward', nbatch, nc, reduce_t, scalar_t, offset_t)
    args = (np_out, np_grd, np_inp, shape, out_stride, grd_stride, inp_stride)
    culaunch(func, numel, args)
    return out


def sym_solve(out, inp, mat, wgt=None, dtype=None):
    """
    Parameters
    ----------
    out : (*batch, C) tensor
    inp : (*batch, C) tensor
    mat : (*batch, C*(C+1)//2) tensor
    wgt : (*batch, C) tensor

    Returns
    -------
    out : (*batch, C) tensor
    """
    nc = inp.shape[-1]
    nbatch = out.ndim - 1
    numel = out.shape[:-1].numel()
    np_inp = to_cupy(inp)
    np_out = to_cupy(out)
    np_mat = to_cupy(mat)

    offset_t = cp.int64
    scalar_t = np_inp.dtype
    reduce_t = dtype or cp.float64

    shape, out_stride = cinfo(np_out, dtype=offset_t, backend=cp)
    _, inp_stride = cinfo(np_inp, dtype=offset_t, backend=cp)
    _, mat_stride = cinfo(np_mat, dtype=offset_t, backend=cp)

    if wgt is not None:
        np_wgt = to_cupy(wgt)
        _, wgt_stride = cinfo(np_wgt, dtype=offset_t, backend=cp)
    else:
        np_wgt = cp.empty([], dtype=scalar_t)
        wgt_stride = cp.empty([], dtype=offset_t)

    # dispatch
    func = kernels.get('sym_solve', nbatch, nc, reduce_t, scalar_t, offset_t)
    args = (np_out, np_inp, np_mat, np_wgt, shape,
            out_stride, inp_stride, mat_stride, wgt_stride)
    culaunch(func, numel, args)
    return out


def sym_solve_(inp, mat, wgt=None, dtype=None):
    """
    Parameters
    ----------
    inp : (*batch, C) tensor
    mat : (*batch, C*(C+1)//2) tensor
    wgt : (*batch, C) tensor

    Returns
    -------
    out : (*batch, C) tensor
    """
    nc = inp.shape[-1]
    nbatch = inp.ndim - 1
    numel = inp.shape[:-1].numel()
    np_inp = to_cupy(inp)
    np_mat = to_cupy(mat)

    offset_t = cp.int64
    scalar_t = np_inp.dtype
    reduce_t = dtype or cp.float64

    shape, inp_stride = cinfo(np_inp, dtype=offset_t, backend=cp)
    _, mat_stride = cinfo(np_mat, dtype=offset_t, backend=cp)

    if wgt is not None:
        np_wgt = wgt.numpy()
        _, wgt_stride = cinfo(np_wgt, dtype=offset_t)
    else:
        np_wgt = cp.empty([], dtype=scalar_t)
        wgt_stride = cp.empty([], dtype=offset_t)

    # dispatch
    func = kernels.get('sym_solve_', nbatch, nc, reduce_t, scalar_t, offset_t)
    args = (np_inp, np_mat, np_wgt, shape, inp_stride, mat_stride, wgt_stride)
    culaunch(func, numel, args)
    return inp


def sym_invert(out, mat, dtype=None):
    """
    Parameters
    ----------
    out : (*batch, C*(C+1)//2) tensor
    mat : (*batch, C*(C+1)//2) tensor

    Returns
    -------
    out : (*batch, C) tensor
    """
    nc = mat.shape[-1]
    nbatch = out.ndim - 1
    numel = out.shape[:-1].numel()
    nc = (int(math.sqrt((1 + 8 * nc))) - 1) // 2
    np_out = to_cupy(out)
    np_mat = to_cupy(mat)

    offset_t = cp.int64
    scalar_t = np_out.dtype
    reduce_t = cp.dtype(dtype or cp.float64)

    shape, out_stride = cinfo(np_out, dtype=offset_t, backend=cp)
    _, mat_stride = cinfo(np_mat, dtype=offset_t, backend=cp)

    # dispatch
    func = kernels.get('sym_invert', nbatch, nc, reduce_t, scalar_t, offset_t)
    args = (np_out, np_mat, shape, out_stride, mat_stride)
    culaunch(func, numel, args)
    return out


def sym_invert_(mat, dtype=None):
    """
    Parameters
    ----------
    mat : (*batch, C*(C+1)//2) tensor

    Returns
    -------
    out : (*batch, C) tensor
    """
    nc = mat.shape[-1]
    nbatch = mat.ndim - 1
    numel = mat.shape[:-1].numel()
    nc = (int(math.sqrt((1 + 8 * nc))) - 1) // 2
    np_mat = to_cupy(mat)

    offset_t = cp.int64
    scalar_t = np_mat.dtype
    reduce_t = cp.dtype(dtype or cp.float64)

    shape, stride = cinfo(np_mat, dtype=offset_t, backend=cp)

    # dispatch
    func = kernels.get('sym_invert_', nbatch, nc, reduce_t, scalar_t, offset_t)
    args = (np_mat, shape, stride)
    culaunch(func, numel, args)
    return mat
