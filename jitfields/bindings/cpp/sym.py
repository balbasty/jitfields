from .utils import cwrap, ctype
from ..common.utils import cinfo, ctypename
import ctypes
import cppyy
import numpy as np
import math
from .utils import include

include()
cppyy.include('posdef.hpp')


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
    np_inp = inp.numpy()
    np_out = out.numpy()
    np_mat = mat.numpy()

    offset_t = np.int64
    scalar_t = np_inp.dtype
    reduce_t = dtype or np.float64

    shape, out_stride = cinfo(np_out, dtype=offset_t)
    _, inp_stride = cinfo(np_inp, dtype=offset_t)
    _, mat_stride = cinfo(np_mat, dtype=offset_t)
    nalldim = int(np_out.ndim)

    # dispatch
    template = f'{nbatch}, {nc}, '
    template += ', '.join(map(ctypename, [reduce_t, scalar_t, offset_t]))
    func = cwrap(func[template])
    func(np_out, np_mat, np_inp, shape, out_stride, mat_stride, inp_stride)
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
    func = cppyy.gbl.jf.posdef.sym_matvec
    return _sym_matvec_base(func, out, inp, mat, dtype)


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
    func = cppyy.gbl.jf.posdef.sym_addmatvec_
    return _sym_matvec_base(func, out, inp, mat, dtype)


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
    func = cppyy.gbl.jf.posdef.sym_submatvec_
    return _sym_matvec_base(func, out, inp, mat, dtype)


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
    np_inp = inp.numpy()
    np_out = out.numpy()
    np_grd = grd.numpy()

    offset_t = np.int64
    scalar_t = np_inp.dtype
    reduce_t = dtype or np.float64

    shape, grd_stride = cinfo(np_grd, dtype=offset_t)
    _, inp_stride = cinfo(np_inp, dtype=offset_t)
    _, out_stride = cinfo(np_out, dtype=offset_t)

    # dispatch
    template = f'{nbatch}, {nc}, '
    template += ', '.join(map(ctypename, [reduce_t, scalar_t, offset_t]))
    func = cwrap(cppyy.gbl.jf.posdef.sym_matvec_backward[template])
    func(np_out, np_grd, np_inp, shape, out_stride, grd_stride, inp_stride)
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
    np_inp = inp.numpy()
    np_out = out.numpy()
    np_mat = mat.numpy()

    offset_t = np.int64
    scalar_t = np_inp.dtype
    reduce_t = dtype or np.float64

    shape, out_stride = cinfo(np_out, dtype=offset_t)
    _, inp_stride = cinfo(np_inp, dtype=offset_t)
    _, mat_stride = cinfo(np_mat, dtype=offset_t)

    if wgt is not None:
        np_wgt = wgt.numpy()
        _, wgt_stride = cinfo(np_wgt, dtype=offset_t)
    else:
        np_wgt = ctypes.POINTER(ctype(scalar_t))()
        wgt_stride = ctypes.POINTER(ctype(offset_t))()

    # dispatch
    template = f'{nbatch}, {nc}, '
    template += ', '.join(map(ctypename, [reduce_t, scalar_t, offset_t]))
    func = cwrap(cppyy.gbl.jf.posdef.sym_solve[template])
    func(np_out, np_inp, np_mat, np_wgt, shape,
         out_stride, inp_stride, mat_stride, wgt_stride)
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
    np_inp = inp.numpy()
    np_mat = mat.numpy()

    offset_t = np.int64
    scalar_t = np_inp.dtype
    reduce_t = dtype or np.float64

    shape, inp_stride = cinfo(np_inp, dtype=offset_t)
    _, mat_stride = cinfo(np_mat, dtype=offset_t)

    if wgt is not None:
        np_wgt = wgt.numpy()
        _, wgt_stride = cinfo(np_wgt, dtype=offset_t)
    else:
        np_wgt = ctypes.POINTER(ctype(scalar_t))()
        wgt_stride = ctypes.POINTER(ctype(offset_t))()

    # dispatch
    template = f'{nbatch}, {nc}, '
    template += ', '.join(map(ctypename, [reduce_t, scalar_t, offset_t]))
    func = cwrap(cppyy.gbl.jf.posdef.sym_solve_[template])
    func(np_inp, np_mat, np_wgt, shape,
         inp_stride, mat_stride, wgt_stride)
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
    nc = (int(math.sqrt((1 + 8 * nc))) - 1) // 2
    np_out = out.numpy()
    np_mat = mat.numpy()

    offset_t = np.int64
    scalar_t = np_out.dtype
    reduce_t = dtype or np.float64

    shape, out_stride = cinfo(np_out, dtype=offset_t)
    _, mat_stride = cinfo(np_mat, dtype=offset_t)

    # dispatch
    template = f'{nbatch}, {nc}, '
    template += ', '.join(map(ctypename, [reduce_t, scalar_t, offset_t]))
    func = cwrap(cppyy.gbl.jf.posdef.sym_invert[template])
    func(np_out, np_mat, shape, out_stride, mat_stride)
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
    nc = (int(math.sqrt((1 + 8 * nc))) - 1) // 2
    np_mat = mat.numpy()

    offset_t = np.int64
    scalar_t = np_mat.dtype
    reduce_t = dtype or np.float64

    shape, stride = cinfo(np_mat, dtype=offset_t)

    # dispatch
    template = f'{nbatch}, {nc}, '
    template += ', '.join(map(ctypename, [reduce_t, scalar_t, offset_t]))
    func = cwrap(cppyy.gbl.jf.posdef.sym_invert_[template])
    func(np_mat, shape, stride)
    return mat
