r"""
This module contains finite-difference regularisers and associated
utilities for tensors that contain dense vector fields.

Each function implement the following set of penalties, which can be
combined:

- **absolute**: the absolute energy is the sum of squared values, _i.e._,
    $\mathcal{L} = \frac{1}{2} \sum_c \int f_c\left(\mathbf{x}\right)^2 \mathrm{d}\mathbf{x}$.
- **membrane**: the membrane energy is the sum of squared first order
    derivatives, _i.e._,
    $\mathcal{L} = \frac{1}{2} \sum_c \int \lVert \left(\boldsymbol{\nabla}f_c\right)\left(\mathbf{x}\right) \rVert_2^2 \mathrm{d}\mathbf{x}$,
    where $\boldsymbol{\nabla}f_c$ is the gradient of the _c_-th component
    of the vector field $\boldsymbol{f}$.
- **bending**: the bending energy is the sum of squared second order
    derivatives, _i.e._,
    $\mathcal{L} = \frac{1}{2} \sum_c \int \lVert \left(\mathbf{H}f_c\right)\left(\mathbf{x}\right) \rVert_F^2 \mathrm{d}\mathbf{x}$,
    where $\mathbf{H}f_c$ is the Hessian matrix of the _c_-th component
    of the vector field $\boldsymbol{f}$.

We further allow a local weighting of the penalty, which in the case
of the membrane energy yields
$\mathcal{L} = \frac{1}{2} \sum_c \int w_c\left(\mathbf{x}\right) \lVert \left(\boldsymbol{\nabla}f_c\right)\left(\mathbf{x}\right) \rVert_2^2 \mathrm{d}\mathbf{x}$.

In practice, $\boldsymbol{f}$ is a dense discrete field, _i.e._, $\mathbf{f} \in \mathbb{R}^{NC}$
(where $N$ is the number of voxels). All these penalties
are then quadratic forms in $\mathbf{f}$: $\mathcal{L} = \frac{1}{2}\mathbf{f}^{T}\mathbf{Lf}$.
The penalties are implemented using finite difference and, in the absence of
local weighting, $\mathbf{L}$ takes the form of a convolution with a small kernel.
"""

__all__ = [
    'field_matvec',
    'field_matvec_add', 'field_matvec_add_',
    'field_matvec_sub', 'field_matvec_sub_',
    'field_kernel',
    'field_kernel_add', 'field_kernel_add_',
    'field_kernel_sub', 'field_kernel_sub_',
    'field_diag',
    'field_diag_add', 'field_diag_add_',
    'field_diag_sub', 'field_diag_sub_',
    'field_precond', 'field_precond_',
    'field_forward',
    'field_relax_',
]

import torch
from torch import Tensor
from typing import Optional
from .utils import try_import, ensure_list, broadcast, make_vector
from .utils import BoundType, OneOrSeveral
from .sym import sym_solve, sym_solve_, sym_matvec
cuda_impl = try_import('jitfields.bindings.cuda', 'regularisers')
cpu_impl = try_import('jitfields.bindings.cpp', 'regularisers')


def field_matvec(
    ndim: int,
    vec: Tensor,
    weight: Optional[Tensor] = None,
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Apply a spatial regularization matrix.

    Notes
    -----
    This function computes the matrix-vector product
    $\mathbf{L} \times \mathbf{f}$, where $\mathbf{f}$ is a scalar or
    vector field and $\mathbf{L}$ encodes a finite-difference penalty.

    Parameters
    ----------
    ndim : `int`
        Number of spatial dimensions
    vec : `(*batch, *spatial, nc) tensor`
        Input vector field, with shape `(*batch, *spatial, nc)`.
    weight : `(*batch, *spatial, nc|1) tensor`, optional
        Weight map, to spatially modulate the regularization.
        With shape `(*batch, *spatial, nc|1)`.
    absolute : `[sequence of] float`
        Penalty on absolute values (per channel).
    membrane : `[sequence of] float`
        Penalty on first derivatives (per channel).
    bending : `[sequence of] float`
        Penalty on second derivatives (per channel).
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    out : `(*batch, *spatial, nc) tensor`, optional
        Output placeholder

    Returns
    -------
    out : `(*batch, *spatial, nc) tensor`
    """
    impl = cuda_impl if vec.is_cuda else cpu_impl

    # broadcast
    if weight is not None:
        vec, weight = broadcast(vec, weight, skip_last=1)
        fn = impl.field_matvec_rls
        weight = [weight]
    else:
        fn = impl.field_matvec
        weight = []

    # allocate output
    if out is None:
        out = torch.empty_like(vec)

    nc = vec.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim).tolist()
    absolute = make_vector(absolute, nc).tolist()
    membrane = make_vector(membrane, nc).tolist()
    bending = make_vector(bending, nc).tolist()

    # forward
    fn(out, vec, *weight, bound, voxel_size,
       absolute, membrane, bending)

    return out


def field_matvec_add(
    ndim: int,
    inp: Tensor,
    vec: Tensor,
    weight: Optional[Tensor] = None,
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
    _sub: bool = False,
) -> Tensor:
    """See `field_matvec`"""
    impl = cuda_impl if vec.is_cuda else cpu_impl

    # broadcast
    inp, vec = broadcast(inp, vec)
    if weight is not None:
        vec, weight = broadcast(vec, weight, skip_last=1)
        inp, vec = broadcast(inp, vec)
        fn = impl.field_matvec_rls
        weight = [weight]
    else:
        fn = impl.field_matvec
        weight = []

    # allocate output
    if out is None:
        out = inp.clone()
    else:
        out.copy(inp)

    nc = vec.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim).tolist()
    absolute = make_vector(absolute, nc).tolist()
    membrane = make_vector(membrane, nc).tolist()
    bending = make_vector(bending, nc).tolist()

    # forward
    fn(out, vec, *weight, bound, voxel_size,
       absolute, membrane, bending,
       'sub' if _sub else 'add')

    return out


def field_matvec_add_(
    ndim: int,
    inp: Tensor,
    vec: Tensor,
    weight: Optional[Tensor] = None,
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    _sub: bool = False,
) -> Tensor:
    """See `field_matvec`"""
    impl = cuda_impl if vec.is_cuda else cpu_impl

    # broadcast
    inp, vec = broadcast(inp, vec)
    if weight is not None:
        vec, weight = broadcast(vec, weight, skip_last=1)
        inp, vec = broadcast(inp, vec)
        fn = impl.field_matvec_rls
        weight = [weight]
    else:
        fn = impl.field_matvec
        weight = []

    nc = vec.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim).tolist()
    absolute = make_vector(absolute, nc).tolist()
    membrane = make_vector(membrane, nc).tolist()
    bending = make_vector(bending, nc).tolist()

    # forward
    fn(inp, vec, *weight, bound, voxel_size,
       absolute, membrane, bending,
       'sub' if _sub else 'add')

    return inp


def field_matvec_sub(
    ndim: int,
    inp: Tensor,
    vec: Tensor,
    weight: Optional[Tensor] = None,
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """See `field_matvec`"""
    return field_matvec_add(ndim, inp, vec, weight,
                             absolute, membrane, bending,
                             bound, voxel_size, out, True)


def field_matvec_sub_(
    ndim: int,
    inp: Tensor,
    vec: Tensor,
    weight: Optional[Tensor] = None,
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
) -> Tensor:
    """See `field_matvec`"""
    return field_matvec_add_(ndim, inp, vec, weight,
                              absolute, membrane, bending,
                              bound, voxel_size, True)


def field_kernel(
    shape: OneOrSeveral[int],
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
    **backend,
) -> Tensor:
    """Return the kernel of a Toeplitz regularization matrix.

    Parameters
    ----------
    shape : `int or list[int]`
        Number of spatial dimensions or shape of the tensor
    absolute : `[sequence of] float`
        Penalty on absolute values (per channel).
    membrane : `[sequence of] float`
        Penalty on first derivatives (per channel).
    bending : `[sequence of] float`
        Penalty on second derivatives (per channel).
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    out : `(*shape, nc) tensor,` optional
        Output placeholder

    Returns
    -------
    out : `(*shape, nc) tensor`
        Convolution kernel, with shape `(*shape, nc)`.
    """
    if isinstance(shape, int):
        ndim = shape
        shape = [5 if bending else 3] * ndim
    else:
        ndim = len(shape)
        shape = list(shape)

    if out is not None:
        nc = out.shape[-1]
    else:
        absolute = ensure_list(absolute)
        membrane = ensure_list(membrane)
        bending = ensure_list(bending)
        nc = max(len(absolute), len(membrane), len(bending))
    shape += [nc]

    # allocate output
    if out is None:
        out = torch.empty(shape, **backend)

    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim).tolist()
    absolute = make_vector(absolute, nc).tolist()
    membrane = make_vector(membrane, nc).tolist()
    bending = make_vector(bending, nc).tolist()

    minshape = (1 if (sum(membrane) == sum(bending) == 0) else
                3 if (sum(bending) == 0) else 5)
    if any(s < minshape for s in out.shape[-ndim-1:-1]):
        raise ValueError(f'Output spatial shape is smaller than the kernel. '
                         f'Spatial shape must be at least {[minshape]*ndim}.')

    # forward
    impl = cuda_impl if out.is_cuda else cpu_impl
    impl.field_kernel(out, bound, voxel_size,
                      absolute, membrane, bending)

    return out


def field_kernel_add(
    ndim: int,
    inp: Tensor,
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
    _sub: bool = False,
) -> Tensor:
    """See `flow_kernel`"""
    # allocate output
    if out is None:
        out = inp.clone()
    else:
        out.copy_(inp)

    nc = inp.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim).tolist()
    absolute = make_vector(absolute, nc).tolist()
    membrane = make_vector(membrane, nc).tolist()
    bending = make_vector(bending, nc).tolist()

    # forward
    impl = cuda_impl if out.is_cuda else cpu_impl
    impl.flow_kernel(out, bound, voxel_size,
                     absolute, membrane, bending,
                     'sub' if _sub else 'add')

    return out


def field_kernel_add_(
    ndim: int,
    inp: Tensor,
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    _sub: bool = False,
) -> Tensor:
    """See `flow_kernel`"""
    nc = inp.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim).tolist()
    absolute = make_vector(absolute, nc).tolist()
    membrane = make_vector(membrane, nc).tolist()
    bending = make_vector(bending, nc).tolist()

    # forward
    impl = cuda_impl if inp.is_cuda else cpu_impl
    impl.flow_kernel(inp, bound, voxel_size,
                     absolute, membrane, bending,
                     'sub' if _sub else 'add')
    return inp


def field_kernel_sub(
    ndim: int,
    inp: Tensor,
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """See `field_kernel`"""
    return field_kernel_add(ndim, inp, absolute, membrane, bending,
                            bound, voxel_size, out)


def field_kernel_sub_(
    ndim: int,
    inp: Tensor,
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
) -> Tensor:
    """See `field_kernel`"""
    return field_kernel_add_(ndim, inp, absolute, membrane, bending,
                             bound, voxel_size)


def field_diag(
    shape: OneOrSeveral[int],
    weight: Optional[Tensor],
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
    **backend,
) -> Tensor:
    """Return the diagonal of a regularization matrix.

    Parameters
    ----------
    shape : `list[int]`
        Shape of the tensor
    weight : `(*batch, *spatial, nc|1) tensor`, optional
        Weight map, to spatially modulate the regularization.
        With shape `(*batch, *spatial, nc|1)`.
    absolute : `[sequence of] float`
        Penalty on absolute values (per channel).
    membrane : `[sequence of] float`
        Penalty on first derivatives (per channel).
    bending : `[sequence of] float`
        Penalty on second derivatives (per channel).
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    out : `(*batch, *spatial, nc) tensor`, optional
        Output placeholder, with shape `(*batch, *spatial, nc)`

    Returns
    -------
    out : `(*batch, *spatial, nc) tensor`
        Diagonal of the regulariser, with shape `(*batch, *spatial, nc)`.
    """
    ndim = len(shape)

    if weight is not None:
        batch = weight.shape[:-ndim-1]
        backend = dict(dtype=weight.dtype, device=weight.device)
        weight = [weight]
    else:
        batch = []
        weight = []

    if out is not None:
        nc = out.shape[-1]
    else:
        absolute = ensure_list(absolute)
        membrane = ensure_list(membrane)
        bending = ensure_list(bending)
        nc = max(len(absolute), len(membrane), len(bending))
        if weight:
            nc = max(nc, weight[0].shape[-1])
        out = torch.empty([*batch, *shape, nc], **backend)

    # forward
    impl = cuda_impl if out.is_cuda else cpu_impl
    fn = impl.field_diag_rls if weight else impl.field_diag

    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim).tolist()
    absolute = make_vector(absolute, nc).tolist()
    membrane = make_vector(membrane, nc).tolist()
    bending = make_vector(bending, nc).tolist()

    fn(out, *weight, bound, voxel_size,
       absolute, membrane, bending)

    return out


def field_diag_add(
    ndim: int,
    inp: Tensor,
    weight: Optional[Tensor],
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
    _sub: bool = False,
) -> Tensor:
    """See `field_diag`"""
    impl = cuda_impl if inp.is_cuda else cpu_impl

    if weight is not None:
        fn = impl.field_diag_rls
        inp, weight = broadcast(inp, weight, skip_last=1)
        weight = [weight]
    else:
        fn = impl.field_diag
        weight = []

    # allocate output
    if out is None:
        out = inp.clone()
    else:
        out.copy_(inp)

    nc = inp.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim).tolist()
    absolute = make_vector(absolute, nc).tolist()
    membrane = make_vector(membrane, nc).tolist()
    bending = make_vector(bending, nc).tolist()

    # forward
    fn(out, *weight, bound, voxel_size,
       absolute, membrane, bending,
       'sub' if _sub else 'add')

    return out


def field_diag_add_(
    ndim: int,
    inp: Tensor,
    weight: Optional[Tensor],
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    _sub: bool = False,
) -> Tensor:
    """See `field_diag`"""
    impl = cuda_impl if inp.is_cuda else cpu_impl

    if weight is not None:
        fn = impl.field_diag_rls
        inp, weight = broadcast(inp, weight, skip_last=1)
        weight = [weight]
    else:
        fn = impl.field_diag
        weight = []

    nc = inp.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim).tolist()
    absolute = make_vector(absolute, nc).tolist()
    membrane = make_vector(membrane, nc).tolist()
    bending = make_vector(bending, nc).tolist()

    # forward
    fn(inp, *weight, bound, voxel_size,
       absolute, membrane, bending,
       'sub' if _sub else 'add')

    return inp


def field_diag_sub(
    ndim: int,
    inp: Tensor,
    weight: Optional[Tensor],
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """See `field_diag`"""
    return field_diag_add(ndim, inp, weight,
                          absolute, membrane, bending,
                          bound, voxel_size, out, True)


def field_diag_sub_(
    ndim: int,
    inp: Tensor,
    weight: Optional[Tensor],
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
) -> Tensor:
    """See `field_diag`"""
    return field_diag_add_(ndim, inp, weight,
                           absolute, membrane, bending,
                           bound, voxel_size, True)


def field_precond(
    ndim: int,
    mat: Tensor,
    vec: Tensor,
    weight: Optional[Tensor],
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Apply the preconditioning
    $(\mathbf{H} + \operatorname{diag}(\mathbf{L}))^{-1} \times \mathbf{f}$.

    Parameters
    ----------
    ndim : `int`
        Number of spatial dimensions
    mat : `(*batch, *spatial, CC) tensor`
        Preconditioning matrix $\mathbf{H}$ with shape `(*batch, *spatial, CC)`,
        where `CC` is one of `{1, C, C*(C+1)//2, C*C}`.
    vec : `(*batch, *spatial, C) tensor`
        Point $\mathbf{f}$ at which to solve the system,
        with shape `(*batch, *spatial, C)`.
    weight : `(*batch, *spatial, nc|1) tensor`, optional
        Regularization weight map, with shape `(*batch, *spatial, nc|1)`.
    absolute : `float`
        Penalty on absolute values.
    membrane : `float`
        Penalty on first derivatives.
    bending : `float`
        Penalty on second derivatives.
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    out : `(*batch, *spatial, C) tensor`
        Output placeholder, with shape `(*batch, *spatial, C)`.

    Returns
    -------
    out : `(*batch, *spatial, C) tensor`
        Preconditioned vector, with shape `(*batch, *spatial, C)`

    """
    shape = vec.shape[-ndim-1:-1]
    diag = field_diag(shape, weight,
                      absolute, membrane, bending,
                      bound, voxel_size)
    return sym_solve(mat, vec, diag, out=out)


def field_precond_(
    ndim: int,
    mat: Tensor,
    vec: Tensor,
    weight: Optional[Tensor],
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
) -> Tensor:
    """See `field_precond`"""
    shape = vec.shape[-ndim-1:-1]
    diag = field_diag(shape, weight,
                      absolute, membrane, bending,
                      bound, voxel_size)
    return sym_solve_(mat, vec, diag)


def field_forward(
    ndim: int,
    mat: Tensor,
    vec: Tensor,
    weight: Optional[Tensor],
    absolute: OneOrSeveral[float] = 0,
    membrane: OneOrSeveral[float] = 0,
    bending: OneOrSeveral[float] = 0,
    bound: OneOrSeveral[BoundType]= 'dct2',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
    ) -> Tensor:
    r"""Apply the forward matrix-vector product
    $(\mathbf{H} + \mathbf{L}) \times \mathbf{f}$.

    Parameters
    ----------
    ndim : `int`
        Number of spatial dimensions
    mat : `(*batch, *spatial, CC) tensor`
        Matrix field $\mathbf{H}$ with shape `(*batch, *spatial, CC)`,
        where `CC` is one of `{1, C, C*(C+1)//2, C*C}`.
    vec : `(*batch, *spatial, C) tensor`
        Vector field $\mathbf{f}$, with shape `(*batch, *spatial, C)`.
    weight : `(*batch, *spatial, nc|1) tensor`, optional
        Regularization weight map, with shape `(*batch, *spatial, nc|1)`.
    absolute : `float`
        Penalty on absolute values.
    membrane : `float`
        Penalty on first derivatives.
    bending : `float`
        Penalty on second derivatives.
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    out : `(*batch, *spatial, C) tensor`
        Output placeholder, with shape `(*batch, *spatial, C)`.

    Returns
    -------
    out : `(*batch, *spatial, C) tensor`
        Matrix-vector product, with shape `(*batch, *spatial, C)`.

    """
    out = sym_matvec(mat, vec, out=out)
    out = field_matvec_add_(ndim, out, vec, weight,
                            absolute, membrane, bending,
                            bound, voxel_size)
    return out


def field_relax_(
        ndim: int,
        field: Tensor,
        hes: Tensor,
        grd: Tensor,
        weight: Optional[Tensor] = None,
        absolute: OneOrSeveral[float] = 0,
        membrane: OneOrSeveral[float] = 0,
        bending: OneOrSeveral[float] = 0,
        bound: OneOrSeveral[BoundType] = 'dct2',
        voxel_size: OneOrSeveral[float] = 1,
        nb_iter: int = 1,
) -> Tensor:
    """Perform relaxation iterations (inplace).

    Parameters
    ----------
    ndim : `int`
        Number of spatial dimensions
    field : `(*batch, *spatial, nc) tensor`
        Warm start, with shape `(*batch, *spatial, nc)`.
    hes : `(*batch, *spatial, nc*(nc+1)//2) tensor`
        Input symmetric Hessian, in voxels.
        With shape `(*batch, *spatial, nc*(nc+1)//2)`.
    grd : `(*batch, *spatial, nc) tensor`
        Input gradient, in voxels.
        With shape `(*batch, *spatial, nc)`.
    weight : `(*batch, *spatial, 1|nc) tensor`, optional
        Weight map, to spatially modulate the regularization.
        With shape `(*batch, *spatial, 1|nc)`.
    absolute : `[sequence of] float`
        Penalty on absolute values.
    membrane : `[sequence of] float`
        Penalty on first derivatives.
    bending : `[sequence of] float`
        Penalty on second derivatives.
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    nb_iter : `int`
        Number of iterations

    Returns
    -------
    flow : `(*batch, *spatial, nc) tensor`
        Refined solution with shape `(*batch, *spatial, nc)` .
    """
    impl = cuda_impl if field.is_cuda else cpu_impl
    nc = field.shape[-1]

    # broadcast
    hes, grd = broadcast(hes, grd, skip_last=1)
    field, grd = broadcast(field, grd, skip_last=1)
    hes, grd = broadcast(hes, grd, skip_last=1)
    if weight is not None:
        flow, weight = broadcast(field, weight, skip_last=1)
        fn = impl.field_relax_rls_
        weight = [weight]
    else:
        fn = impl.field_relax_
        weight = []

    absolute = make_vector(absolute, nc).tolist()
    membrane = make_vector(membrane, nc).tolist()
    bending = make_vector(bending, nc).tolist()

    if sum(membrane) == sum(bending) == 0:
        # closed form in one iteration: x = (H + R) \ g
        if absolute == 0:
            field = sym_solve(hes, grd, *weight, out=field)
        elif weight:
            field = sym_solve(hes, grd, weight[0] + absolute, out=field)
        else:
            field = sym_solve(hes, grd, out=field)
        return field

    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim).tolist()

    # forward
    fn(field, hes, grd, *weight, nb_iter, bound, voxel_size,
       absolute, membrane, bending)

    return field