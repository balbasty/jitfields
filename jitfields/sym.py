"""
## Overview
This module contains linear-algebra routines (matrix-vector product,
matrix inversion, etc.) for batches of symmetric matrices stored in a
compact way (that is, only $N(N+1)/2$ values are stored, instead of $N^2$).

Our compact representation differs from classical "columns" or "rows"
layouts. The compact flattened matrix should contain the diagonal of
the matrix first, followed by the rows of the upper triangle of the
matrix, i.e.:

    [ a d e ]
    [ . b f ]   =>  [a b c d e f]
    [ . . c ]

Note that matrix-vector functions (`matvec`, `solve`) also accept (and
automatically detect) compact diagonal matrices and compact scaled identity
matrices. If the vector has shape `(*, N)` and the matrix has shape `(*, NN)`,
where `*` is any number of leading batch dimensions, `NN` can take values:

- `1`: and the matrix is assumed to be a scaled identity,
- `N`: and the matrix is assumed to be a diagonal matrix,
- `N*(N+1)//2`: and the matrix is assumed to be symmetric,
- `N*N`: and the matrix is assumed to be full.
"""
__all__ = [
    'sym_matvec',
    'sym_addmatvec', 'sym_addmatvec_',
    'sym_submatvec', 'sym_submatvec_',
    'sym_solve', 'sym_solve_',
    'sym_invert', 'sym_invert_'
]

import torch
from torch import Tensor
from typing import Optional
from .utils import try_import, broadcast
cuda_sym = try_import('jitfields.bindings.cuda', 'sym')
cpu_sym = try_import('jitfields.bindings.cpp', 'sym')


def sym_matvec(
    mat: Tensor,
    vec: Tensor,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Matrix-vector product for compact symmetric matrices

    Equivalent to `out = mat @ vec`.

    Parameters
    ----------
    mat : `(..., CC) tensor`
        Symmetric matrix with compact storage, with shape `(..., CC)`,
        where `CC` is one of `{1, C, C*(C+1)//2, C*C}`.

        - If `CC == 1`, the matrix is a scaled identity.
        - If `CC == C`, the matrix is diagonal.
        - If `CC == C*(C+1)//2`, the matrix should be saved as a vector
        containing the diagonal followed by the rows of the upper triangle.
        - If `CC == C*C`, the matrix is a flattened dense matrix.
    vec : `(..., C) tensor`
        Vector with shape `(..., C)`.
    dtype : `torch.dtype`, optional
        Data type used to carry the computation. By default, same as input.
    out : `(..., C) tensor`, optional
        Output placeholder, with shape `(..., C)`.

    Returns
    -------
    out : (..., C) tensor
        Matrix-vector product, with shape `(..., C)`.

    """
    nc = vec.shape[-1]
    nc2 = mat.shape[-1]
    mat, vec = _broadcast_matvec(mat, vec)
    out = _allocate_out(vec, out)
    if nc2 == (nc*(nc+1))//2:
        return MatVec.apply(mat, vec, dtype, out)
    elif nc2 == nc*nc:
        if out is not None:
            out = out.unsqueeze(-1)
        mat = mat.reshape([*mat.shape[:-1], nc, nc])
        return torch.matmul(mat, vec.unsqueeze(-1), out=out).squeeze(-1)
    else:
        return torch.mul(mat, vec, out=out)


def sym_addmatvec(
    inp: Tensor,
    mat: Tensor,
    vec: Tensor,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Add a matrix-vector product for compact symmetric matrices

    Equivalent to `out = inp + mat @ vec`

    Parameters
    ----------
    inp : `(..., C) tensor`
        Vector to which the matrix-vector product is added.
        With shape `(..., C)`.
    mat : `(..., CC) tensor`
        Symmetric matrix with compact storage, with shape `(..., CC)`,
        where `CC` is one of `{1, C, C*(C+1)//2, C*C}`.

        - If `CC == 1`, the matrix is a scaled identity.
        - If `CC == C`, the matrix is diagonal.
        - If `CC == C*(C+1)//2`, the matrix should be saved as a vector
        containing the diagonal followed by the rows of the upper triangle.
        - If `CC == C*C`, the matrix is a flattened dense matrix.
    vec : `(..., C) tensor`
        Vector with shape `(..., C)`.
    dtype : `torch.dtype`, optional
        Data type used to carry the computation. By default, same as input.
    out : `(..., C) tensor`, optional
        Output placeholder, with shape `(..., C)`.


    Returns
    -------
    out : `(..., C) tensor`
        Added matrix-vector product, with shape `(..., C)`.

    """
    nc = vec.shape[-1]
    nc2 = mat.shape[-1]
    inp, vec = _broadcast_vecvec(inp, vec)
    mat, vec = _broadcast_matvec(mat, vec)
    mat, inp = _broadcast_matvec(mat, inp)
    out = _allocate_out(inp, out)
    if nc2 == (nc*(nc+1))//2:
        return AddMatVec.apply(inp, mat, vec, dtype, out)
    elif nc2 == nc*nc:
        if out is not None:
            out = out.unsqueeze(-1)
        return torch.matmul(
            mat, vec.unsqueeze(-1), out=out).squeeze(-1).add_(inp)
    else:
        return torch.mul(mat, vec, out=out).add_(inp)


def sym_addmatvec_(
    inp: Tensor,
    mat: Tensor,
    vec: Tensor,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Inplace add a matrix-vector product for compact symmetric matrices

    Equivalent to `inp += mat @ vec`

    Parameters
    ----------
    inp : `(..., C) tensor`
        Vector to which the matrix-vector product is added.
        With shape `(..., C)`.
    mat : `(..., CC) tensor`
        Symmetric matrix with compact storage, with shape `(..., CC)`,
        where `CC` is one of `{1, C, C*(C+1)//2, C*C}`.

        - If `CC == 1`, the matrix is a scaled identity.
        - If `CC == C`, the matrix is diagonal.
        - If `CC == C*(C+1)//2`, the matrix should be saved as a vector
        containing the diagonal followed by the rows of the upper triangle.
        - If `CC == C*C`, the matrix is a flattened dense matrix.
    vec : `(..., C) tensor`
        Vector with shape `(..., C)`.
    dtype : `torch.dtype`, optional
        Data type used to carry the computation. By default, same as input.


    Returns
    -------
    inp : `(..., C) tensor`
        Added matrix-vector product, with shape `(..., C)`.

    """
    nc = vec.shape[-1]
    nc2 = mat.shape[-1]
    inp, vec = _broadcast_vecvec(inp, vec)
    mat, vec = _broadcast_matvec(mat, vec)
    mat, inp = _broadcast_matvec(mat, inp)
    if nc2 == (nc*(nc+1))//2:
        return AddMatVec_.apply(inp, mat, vec, dtype)
    elif nc2 == nc*nc:
        return inp.add_(torch.matmul(mat, vec.unsqueeze(-1)))
    else:
        return inp.addcmul_(mat, vec)


def sym_submatvec(
    inp: Tensor,
    mat: Tensor,
    vec: Tensor,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Subtract a matrix-vector product for compact symmetric matrices

    Equivalent to `out = inp - mat @ vec`

    Parameters
    ----------
    inp : `(..., C) tensor`
        Vector to which the matrix-vector product is added.
        With shape `(..., C)`.
    mat : `(..., CC) tensor`
        Symmetric matrix with compact storage, with shape `(..., CC)`,
        where `CC` is one of `{1, C, C*(C+1)//2, C*C}`.

        - If `CC == 1`, the matrix is a scaled identity.
        - If `CC == C`, the matrix is diagonal.
        - If `CC == C*(C+1)//2`, the matrix should be saved as a vector
        containing the diagonal followed by the rows of the upper triangle.
        - If `CC == C*C`, the matrix is a flattened dense matrix.
    vec : `(..., C) tensor`
        Vector with shape `(..., C)`.
    dtype : `torch.dtype`, optional
        Data type used to carry the computation. By default, same as input.
    out : `(..., C) tensor`, optional
        Output placeholder, with shape `(..., C)`.


    Returns
    -------
    out : `(..., C) tensor`
        Subtracted matrix-vector product, with shape `(..., C)`.

    """
    nc = vec.shape[-1]
    nc2 = mat.shape[-1]
    inp, vec = _broadcast_vecvec(inp, vec)
    mat, vec = _broadcast_matvec(mat, vec)
    mat, inp = _broadcast_matvec(mat, inp)
    out = _allocate_out(inp, out)
    if nc2 == (nc*(nc+1))//2:
        return SubMatVec.apply(inp, mat, vec, dtype, out)
    elif nc2 == nc*nc:
        if out is not None:
            out = out.unsqueeze(-1)
        return torch.matmul(
            mat, vec.unsqueeze(-1), out=out).squeeze(-1).neg_().add_(inp)
    else:
        return torch.mul(mat, vec, out=out).neg_().add_(inp)


def sym_submatvec_(
    inp: Tensor,
    mat: Tensor,
    vec: Tensor,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Inplace subtract a matrix-vector product for compact symmetric matrices

    Equivalent to `inp -= mat @ vec`

    Parameters
    ----------
    inp : `(..., C) tensor`
        Vector to which the matrix-vector product is added.
        With shape `(..., C)`.
    mat : `(..., CC) tensor`
        Symmetric matrix with compact storage, with shape `(..., CC)`,
        where `CC` is one of `{1, C, C*(C+1)//2, C*C}`.

        - If `CC == 1`, the matrix is a scaled identity.
        - If `CC == C`, the matrix is diagonal.
        - If `CC == C*(C+1)//2`, the matrix should be saved as a vector
        containing the diagonal followed by the rows of the upper triangle.
        - If `CC == C*C`, the matrix is a flattened dense matrix.
    vec : `(..., C) tensor`
        Vector with shape `(..., C)`.
    dtype : `torch.dtype`, optional
        Data type used to carry the computation. By default, same as input.


    Returns
    -------
    inp : `(..., C) tensor`
        Subtracted matrix-vector product, with shape `(..., C)`.

    """
    nc = vec.shape[-1]
    nc2 = mat.shape[-1]
    inp, vec = _broadcast_vecvec(inp, vec)
    mat, vec = _broadcast_matvec(mat, vec)
    mat, inp = _broadcast_matvec(mat, inp)
    if nc2 == (nc*(nc+1))//2:
        return SubMatVec_.apply(inp, mat, vec, dtype)
    elif nc2 == nc*nc:
        return inp.sub_(torch.matmul(mat, vec.unsqueeze(-1)))
    else:
        return inp.addcmul_(mat, vec, value=-1)


def sym_solve(mat, vec, diag=None, dtype=None, out=None):
    """Solve the symmetric linear system

    Equivalent to `out = (mat + diag).inverse() @ vec`

    !!! warning
        Does not backpropagate through `mat`.

    Parameters
    ----------
    mat : `(..., CC) tensor`
        Symmetric matrix with compact storage, with shape `(..., CC)`,
        where `CC` is one of `{1, C, C*(C+1)//2, C*C}`.

        - If `CC == 1`, the matrix is a scaled identity.
        - If `CC == C`, the matrix is diagonal.
        - If `CC == C*(C+1)//2`, the matrix should be saved as a vector
        containing the diagonal followed by the rows of the upper triangle.
        - If `CC == C*C`, the matrix is a flattened dense matrix.
    vec : `(..., C) tensor`
        Vector with shape `(..., C)`.
    diag : `(..., C) tensor`
        Diagonal regularizer, with shape `(..., C)`.
    dtype : `torch.dtype`, optional
        Data type used to carry the computation. By default, same as input.
    out : `(..., C) tensor`, optional
        Output placeholder, with shape `(..., C)`.

    Returns
    -------
    out : `(..., C) tensor`
        Solution of the linear system, with shape `(..., C)`.

    """
    nc = vec.shape[-1]
    nc2 = mat.shape[-1]
    mat, vec = _broadcast_matvec(mat, vec)
    out = _allocate_out(vec, out)
    if nc2 == (nc*(nc+1))//2:
        return Solve.apply(mat, vec, diag, dtype, out)
    elif nc2 == nc*nc:
        if out is not None:
            out = out.unsqueeze(-1)
        return torch.solve(mat, vec.unsqueeze(-1), out=out).squeeze(-1)
    else:
        return torch.div(vec, mat, out=out)


def sym_solve_(mat, vec, diag=None, dtype=None):
    """Solve the symmetric linear system in-place

    Equivalent to `vec = mat.inverse() @ vec`

    !!! warning
        Does not backpropagate through `mat`.

    Parameters
    ----------
    mat : `(..., CC) tensor`
        Symmetric matrix with compact storage, with shape `(..., CC)`,
        where `CC` is one of `{1, C, C*(C+1)//2, C*C}`.

        - If `CC == 1`, the matrix is a scaled identity.
        - If `CC == C`, the matrix is diagonal.
        - If `CC == C*(C+1)//2`, the matrix should be saved as a vector
        containing the diagonal followed by the rows of the upper triangle.
        - If `CC == C*C`, the matrix is a flattened dense matrix.
    vec : `(..., C) tensor`
        Vector with shape `(..., C)`.
    diag : `(..., C) tensor`
        Diagonal regularizer, with shape `(..., C)`.
    dtype : `torch.dtype`, optional
        Data type used to carry the computation. By default, same as input.

    Returns
    -------
    vec : (..., C) tensor
        Solution of the linear system, with shape `(..., C)`.

    """
    nc = vec.shape[-1]
    nc2 = mat.shape[-1]
    mat, vec = _broadcast_matvec(mat, vec)
    if nc2 == (nc*(nc+1))//2:
        return Solve_.apply(mat, vec, diag, dtype)
    elif nc2 == nc*nc:
        sol = torch.solve(mat, vec.unsqueeze(-1)).squeeze(-1)
        return vec.copy_(sol)
    else:
        return vec.div_(mat)


def sym_invert(mat, dtype=None, out=None):
    """Invert a compact symmetric matrix

    Equivalent to `out = mat.inverse()`

    !!! warning
        Does not backpropagate through `mat`.

    Parameters
    ----------
    mat : `(..., C*(C+1)//2) tensor`
        Symmetric matrix with compact storage, with shape `(..., C*(C+1)//2)`.
        The matrix should be saved as a vector containing the diagonal
        followed by the rows of the upper triangle.
    dtype : torch.dtype, optional
        Data type used to carry the computation. By default, same as input.
    out : `(..., C*(C+1)//2) tensor`, optional
        Output placeholder, with shape `(..., C*(C+1)//2)`.

    Returns
    -------
    mat : `(..., C*(C+1)//2) tensor`
        Inverse matrix, with shape `(..., C*(C+1)//2)`.

    """
    if mat.requires_grad:
        raise ValueError('sym_invert does not backpropagate gradients '
                         'through the matrix. Use `detach()`.')
    out = _allocate_out(mat, out)
    _fwd = cuda_sym.sym_invert if mat.is_cuda else cpu_sym.sym_invert
    return _fwd(out, mat, dtype=dtype)


def sym_invert_(mat, dtype=None):
    """Invert a compact symmetric matrix in-place

    Equivalent to `mat = mat.inverse()`

    !!! warning
        Does not backpropagate through `mat`.

    Parameters
    ----------
    mat : `(..., C*(C+1)//2) tensor`
        Symmetric matrix with compact storage, with shape `(..., C*(C+1)//2)`.
        The matrix should be saved as a vector containing the diagonal
        followed by the rows of the upper triangle.
    dtype : `torch.dtype`, optional
        Data type used to carry the computation. By default, same as input.

    Returns
    -------
    mat : `(..., C*(C+1)//2) tensor`
        Inverse matrix, with shape `(..., C*(C+1)//2)`.

    """
    if mat.requires_grad:
        raise ValueError('sym_invert_ does not backpropagate gradients '
                         'through the matrix. Use `detach()`.')
    _fwd = cuda_sym.sym_invert_ if mat.is_cuda else cpu_sym.sym_invert_
    return _fwd(mat, dtype=dtype)


class MatVec(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mat, vec, dtype, out):
        _fwd = cuda_sym.sym_matvec if mat.is_cuda else cpu_sym.sym_matvec
        ctx.dtype = dtype
        ctx.save_for_backward(mat, vec)
        return _fwd(out, vec, mat, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        mat, vec = ctx.saved_tensors
        _vbwd = cuda_sym.sym_matvec if mat.is_cuda else cpu_sym.sym_matvec
        _mbwd = (cuda_sym.sym_matvec_backward if mat.is_cuda else
                 cpu_sym.sym_matvec_backward)
        gvec = _vbwd(torch.empty_like(vec), mat, grad, dtype=ctx.dtype)
        gmat = _mbwd(torch.empty_like(mat), grad, vec, dtype=ctx.dtype)
        return gmat, gvec, None, None


class AddMatVec(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, mat, vec, dtype, out):
        _fwd = (cuda_sym.sym_addmatvec_ if mat.is_cuda else
                cpu_sym.sym_addmatvec_)
        ctx.dtype = dtype
        ctx.save_for_backward(mat, vec)
        out.copy_(inp)
        return _fwd(out, vec, mat, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        mat, vec = ctx.saved_tensors
        _vbwd = (cuda_sym.sym_addmatvec_ if mat.is_cuda else
                 cpu_sym.sym_addmatvec_)
        _mbwd = (cuda_sym.sym_matvec_backward if mat.is_cuda else
                 cpu_sym.sym_matvec_backward)
        gvec = _vbwd(torch.empty_like(vec), mat, grad, dtype=ctx.dtype)
        gmat = _mbwd(torch.empty_like(mat), grad, vec, dtype=ctx.dtype)
        return grad, gmat, gvec, None, None


class AddMatVec_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, mat, vec, dtype):
        _fwd = (cuda_sym.sym_addmatvec_ if mat.is_cuda else
                cpu_sym.sym_addmatvec_)
        ctx.dtype = dtype
        ctx.save_for_backward(mat, vec)
        return _fwd(inp, vec, mat, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        mat, vec = ctx.saved_tensors
        _vbwd = (cuda_sym.sym_addmatvec_ if mat.is_cuda else
                 cpu_sym.sym_addmatvec_)
        _mbwd = (cuda_sym.sym_matvec_backward if mat.is_cuda else
                 cpu_sym.sym_matvec_backward)
        gvec = _vbwd(torch.empty_like(vec), mat, grad, dtype=ctx.dtype)
        gmat = _mbwd(torch.empty_like(mat), grad, vec, dtype=ctx.dtype)
        return grad, gmat, gvec, None, None


class SubMatVec(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, mat, vec, dtype, out):
        _fwd = (cuda_sym.sym_submatvec_ if mat.is_cuda else
                cpu_sym.sym_submatvec_)
        ctx.dtype = dtype
        ctx.save_for_backward(mat, vec)
        out.copy_(inp)
        return _fwd(out, vec, mat, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        mat, vec = ctx.saved_tensors
        _vbwd = (cuda_sym.sym_submatvec_ if mat.is_cuda else
                 cpu_sym.sym_submatvec_)
        _mbwd = (cuda_sym.sym_matvec_backward if mat.is_cuda else
                 cpu_sym.sym_matvec_backward)
        gvec = _vbwd(torch.empty_like(vec), mat, grad, dtype=ctx.dtype).neg_()
        gmat = _mbwd(torch.empty_like(mat), grad, vec, dtype=ctx.dtype).neg_()
        return grad, gmat, gvec, None, None


class SubMatVec_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, mat, vec, dtype):
        _fwd = (cuda_sym.sym_submatvec_ if mat.is_cuda else
                cpu_sym.sym_submatvec_)
        ctx.dtype = dtype
        ctx.save_for_backward(mat, vec)
        return _fwd(inp, vec, mat, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        mat, vec = ctx.saved_tensors
        _vbwd = (cuda_sym.sym_submatvec_ if mat.is_cuda else
                 cpu_sym.sym_submatvec_)
        _mbwd = (cuda_sym.sym_matvec_backward if mat.is_cuda else
                 cpu_sym.sym_matvec_backward)
        gvec = _vbwd(torch.empty_like(vec), mat, grad, dtype=ctx.dtype).neg_()
        gmat = _mbwd(torch.empty_like(mat), grad, vec, dtype=ctx.dtype).neg_()
        return grad, gmat, gvec, None, None


class Solve(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mat, vec, diag, dtype, out):
        _fwd = cuda_sym.sym_solve if mat.is_cuda else cpu_sym.sym_solve
        ctx.dtype = dtype
        ctx.shape = vec.shape
        ctx.save_for_backward(*([mat, diag] if diag is None else [mat]))
        if mat.requires_grad:
            raise ValueError('sym_solve does not backpropagate gradients '
                             'through the matrix. Use `detach()`.')
        return _fwd(out, vec, mat, diag, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        mat, *diag, = ctx.saved_tensors
        _vbwd = cuda_sym.sym_solve if mat.is_cuda else cpu_sym.sym_solve
        gvec = _vbwd(
            mat.new_empty(ctx.shape), mat, grad, *diag, dtype=ctx.dtype)
        return None, gvec, None, None


class Solve_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mat, vec, diag, dtype):
        _fwd = cuda_sym.sym_solve_ if mat.is_cuda else cpu_sym.sym_solve_
        ctx.dtype = dtype
        ctx.shape = vec.shape
        ctx.save_for_backward(*([mat, diag] if diag is None else [mat]))
        if mat.requires_grad:
            raise ValueError('sym_solve_ does not backpropagate gradients '
                             'through the matrix. Use `detach()`.')
        return _fwd(vec, mat, dtype)

    @staticmethod
    def backward(ctx, grad):
        mat, *diag, = ctx.saved_tensors
        _vbwd = cuda_sym.sym_solve if mat.is_cuda else cpu_sym.sym_solve
        gvec = _vbwd(mat.new_empty(ctx.shape), mat, grad, *diag, ctx.dtype)
        return None, gvec, None, None


def _broadcast_matvec(mat, vec):
    nk = mat.shape[-1]
    nc = vec.shape[-1]
    if nk != (nc*(nc+1))//2:
        raise ValueError(f'Shapes not compatible. Expected a compact matrix '
                         f'of length {(nc*(nc+1))//2} for a vector or length '
                         f'{nc} but got {nk}.')

    return broadcast(mat, vec, 1)


def _broadcast_vecvec(inp, vec):
    nc1 = inp.shape[-1]
    nc2 = vec.shape[-1]
    if nc1 != nc2:
        raise ValueError(f'Shapes not compatible. Expected two vectors '
                         f'of same length but got {nc1} and {nc2}.')
    return broadcast(inp, vec, 1)


def _allocate_out(vec, out):
    if out is None:
        out = torch.empty_like(vec)
    else:
        if out.shape != vec.shape:
            raise ValueError(f'Output shape not compatible. '
                             f'Expected {tuple(vec.shape)} but got '
                             f'{tuple(out.shape)}')
        if not all(out.stride()):
            raise ValueError('Output should not wrap over itself '
                             '(i.e., it should not have zero strides).')
    return out
