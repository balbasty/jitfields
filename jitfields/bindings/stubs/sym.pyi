from torch import Tensor
import torch
import numpy as np
from typing import Optional, Union

DType = Optional[Union[torch.dtype, np.dtype, str]]


def sym_matvec(out: Tensor, inp: Tensor, mat: Tensor,
               dtype: DType = None) -> Tensor:
    """
    Matrix-vector product for symmetric matrices

    Parameters
    ----------
    out : (*batch, C) tensor
    inp : (*batch, C) tensor
    mat : (*batch, C*(C+1)//2) tensor
    dtype : dtype

    Returns
    -------
    out : (*batch, C) tensor
    """
    ...


def sym_addmatvec_(out: Tensor, inp: Tensor, mat: Tensor,
                   dtype: DType = None) -> Tensor:
    """
    Matrix-vector product for symmetric matrices.
    Results gets added to `out`.

    Parameters
    ----------
    out : (*batch, C) tensor
    inp : (*batch, C) tensor
    mat : (*batch, C*(C+1)//2) tensor
    dtype : dtype

    Returns
    -------
    out : (*batch, C) tensor
    """
    ...


def sym_submatvec_(out: Tensor, inp: Tensor, mat: Tensor,
                   dtype: DType = None) -> Tensor:
    """
    Matrix-vector product for symmetric matrices.
    Results gets subtracted from `out`.

    Parameters
    ----------
    out : (*batch, C) tensor
    inp : (*batch, C) tensor
    mat : (*batch, C*(C+1)//2) tensor
    dtype : dtype

    Returns
    -------
    out : (*batch, C) tensor
    """
    ...


def sym_matvec_backward(out: Tensor, grd: Tensor, inp: Tensor,
                        dtype: DType = None) -> Tensor:
    """
    Backward pass of matvec wrt the matrix

    Parameters
    ----------
    out : (*batch, C*(C+1)//2)) tensor
    grd : (*batch, C) tensor
    inp : (*batch, C) tensor
    dtype : dtype

    Returns
    -------
    out : (*batch, C*(C+1)//2)) tensor
    """
    ...


def sym_solve(out: Tensor, inp: Tensor, mat: Tensor,
              wgt: Optional[Tensor] = None,
              dtype: DType = None) -> Tensor:
    r"""
    Solve the linear system `(mat + diag(wgt)) \ inp`

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
    ...


def sym_solve_(inp: Tensor, mat: Tensor, wgt: Optional[Tensor] = None,
               dtype: DType = None) -> Tensor:
    r"""
    Solve the linear system `(mat + diag(wgt)) \ inp` inplace

    Parameters
    ----------
    inp : (*batch, C) tensor
    mat : (*batch, C*(C+1)//2) tensor
    wgt : (*batch, C) tensor

    Returns
    -------
    out : (*batch, C) tensor
    """
    ...


def sym_invert(out: Tensor, mat: Tensor, dtype: DType = None) -> Tensor:
    """
    Invert a symmetric matrix

    Parameters
    ----------
    out : (*batch, C*(C+1)//2) tensor
    mat : (*batch, C*(C+1)//2) tensor

    Returns
    -------
    out : (*batch, C) tensor
    """
    ...


def sym_invert_(mat: Tensor, dtype: DType = None) -> Tensor:
    """
    Invert a symmetric matrix inplace

    Parameters
    ----------
    mat : (*batch, C*(C+1)//2) tensor

    Returns
    -------
    out : (*batch, C) tensor
    """
    ...
