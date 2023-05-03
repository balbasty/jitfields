r"""
This module contains finite-difference regularisers and associated
utilities for tensors that contain dense displacement fields.

Displacements are expected to be expressed in voxels, and relate to the
tensor's lattice. However, the voxel size of the lattice (and therefore,
the size of the displacement) can be provided.

Each function implement the following set of penalties, which can be
combined:

- **absolute**: the absolute energy is the sum of squared values, _i.e._,
    $\mathcal{L} = \frac{1}{2} \int \lVert \boldsymbol{f}\left(\mathbf{x}\right) \rVert_2^2 \mathrm{d}\mathbf{x}$.
- **membrane**: the membrane energy is the sum of squared first order
    derivatives, _i.e._,
    $\mathcal{L} = \frac{1}{2} \int \lVert \left(\mathbf{D}\boldsymbol{f}\right)\left(\mathbf{x}\right) \rVert_F^2 \mathrm{d}\mathbf{x}$,
    where $\mathbf{D}\boldsymbol{f}$ is the Jacobian matrix of the displacement $\boldsymbol{f}$.
- **bending**: the bending energy is the sum of squared second order
    derivatives, _i.e._,
    $\mathcal{L} = \frac{1}{2} \int \sum_d \lVert \left(\mathbf{H}f_d\right)\left(\mathbf{x}\right) \rVert_F^2 \mathrm{d}\mathbf{x}$,
    where $\mathbf{H}f_d$ is the Hessian matrix of the _d_-th component
    of the displacement $\boldsymbol{f}$.
- **shears**: the second Lame component of the linear-elastic energy penalizes
    the norm of the symmetric part of the Jacobian, _i.e._,
    $\mathcal{L} = \frac{1}{2} \int \lVert \frac{1}{2} \left(\mathbf{D}\boldsymbol{f} + \mathbf{D}\boldsymbol{f}^{T}\right) \left(\mathbf{x}\right) \rVert_F^2 \mathrm{d}\mathbf{x}$.
- **div**: the first Lame component of the linear-elastic energy penalizes
    divergence, which is the square of the trace of the Jacobian, _i.e._
    $\mathcal{L} = \frac{1}{2} \int \operatorname{tr}\left(\mathbf{D}\boldsymbol{f}\right)^2 \left(\mathbf{x}\right) \mathrm{d}\mathbf{x}$.

We further allow a local weighting of the penalty, which in the case
of the membrane energy yields
$\mathcal{L} = \frac{1}{2} \int w\left(\mathbf{x}\right) \lVert \left(\mathbf{D}\boldsymbol{f}\right)\left(\mathbf{x}\right) \rVert_F^2 \mathrm{d}\mathbf{x}$.

In practice, $\boldsymbol{f}$ is a dense discrete field, _i.e._, $\mathbf{f} \in \mathbb{R}^{ND}$
(where $N$ is the number of voxels). All these penalties
are then quadratic forms in $\mathbf{f}$: $\mathcal{L} = \frac{1}{2}\mathbf{f}^{T}\mathbf{Lf}$.
The penalties are implemented using finite difference and, in the absence of
local weighting, $\mathbf{L}$ takes the form of a convolution with a small kernel.
"""

__all__ = [
    'flow_matvec',
    'flow_matvec_add', 'flow_matvec_add_',
    'flow_matvec_sub', 'flow_matvec_sub_',
    'flow_kernel',
    'flow_kernel_add', 'flow_kernel_add_',
    'flow_kernel_sub', 'flow_kernel_sub_',
    'flow_diag',
    'flow_diag_add', 'flow_diag_add_',
    'flow_diag_sub', 'flow_diag_sub_',
    'flow_precond', 'flow_precond_',
    'flow_forward',
    'flow_relax_',
]

import torch
from torch import Tensor
from typing import Optional
from .utils import try_import, ensure_list, broadcast
from .utils import BoundType, OneOrSeveral
from .sym import sym_solve, sym_solve_, sym_matvec
cuda_impl = try_import('jitfields.bindings.cuda', 'regularisers')
cpu_impl = try_import('jitfields.bindings.cpp', 'regularisers')


def flow_matvec(
    flow: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Apply a spatial regularization matrix.

    Notes
    -----
    This function computes the matrix-vector product $\mathbf{L} \times \mathbf{f}$,
    where $\mathbf{f}$ is a displacement field (in voxels) and $\mathbf{L}$
    encodes a finite-difference penalty.

    Parameters
    ----------
    flow : `(*batch, *spatial, ndim) tensor`
        Input displacement field, in voxels.
        With shape `(*batch, *spatial, ndim)`.
    weight : `(*batch, *spatial) tensor`, optional
        Weight map, to spatially modulate the regularization.
        With shape `(*batch, *spatial)`.
    absolute : `float`
        Penalty on absolute values.
    membrane : `float`
        Penalty on first derivatives.
    bending : `float`
        Penalty on second derivatives.
    shears : `float`
        Penalty on local shears.
    div : `float`
        Penalty on local volume changes.
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    out : `(*batch, *spatial, ndim) tensor`, optional
        Output placeholder. With shape `(*batch, *spatial, ndim)`.

    Returns
    -------
    out : `(*batch, *spatial, ndim) tensor`
        Matrix vector product, with shape `(*batch, *spatial, ndim)`.
    """
    impl = cuda_impl if flow.is_cuda else cpu_impl

    # broadcast
    if weight is not None:
        flow, weight = broadcast(flow, weight[..., None], skip_last=1)
        weight = weight[..., 0]
        fn = impl.flow_matvec_rls
        weight = [weight]
    else:
        fn = impl.flow_matvec
        weight = []

    # allocate output
    if out is None:
        out = torch.empty_like(flow)

    bound = ensure_list(bound, flow.shape[-1])
    voxel_size = ensure_list(voxel_size, flow.shape[-1])

    # forward
    fn(out, flow, *weight, bound, voxel_size,
       absolute, membrane, bending, shears, div)

    return out


def flow_matvec_add(
    inp: Tensor,
    flow: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
    _sub: bool = False,
) -> Tensor:
    """
    Add the output of [`flow_matvec`](index.html#jitfields.regularisers.flow_matvec)
    to `inp`.
    """
    impl = cuda_impl if flow.is_cuda else cpu_impl

    # broadcast
    inp, flow = broadcast(inp, flow)
    if weight is not None:
        flow, weight = broadcast(flow, weight[..., None], skip_last=1)
        inp, flow = broadcast(inp, flow)
        weight = weight[..., 0]
        fn = impl.flow_matvec_rls
        weight = [weight]
    else:
        fn = impl.flow_matvec
        weight = []

    # allocate output
    if out is None:
        out = inp.clone()
    else:
        out.copy(inp)

    bound = ensure_list(bound, flow.shape[-1])
    voxel_size = ensure_list(voxel_size, flow.shape[-1])

    # forward
    fn(out, flow, *weight, bound, voxel_size,
       absolute, membrane, bending, shears, div,
       'sub' if _sub else 'add')

    return out


def flow_matvec_add_(
    inp: Tensor,
    flow: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    _sub: bool = False,
) -> Tensor:
    """
    Add the output of [`flow_matvec`](index.html#jitfields.regularisers.flow_matvec)
    to `inp` (inplace).
    """
    impl = cuda_impl if flow.is_cuda else cpu_impl

    # broadcast
    inp, flow = broadcast(inp, flow)
    if weight is not None:
        flow, weight = broadcast(flow, weight[..., None], skip_last=1)
        inp, flow = broadcast(inp, flow)
        weight = weight[..., 0]
        fn = impl.flow_matvec_rls
        weight = [weight]
    else:
        fn = impl.flow_matvec
        weight = []

    bound = ensure_list(bound, flow.shape[-1])
    voxel_size = ensure_list(voxel_size, flow.shape[-1])

    # forward
    fn(inp, flow, *weight, bound, voxel_size,
       absolute, membrane, bending, shears, div,
       'sub' if _sub else 'add')

    return inp


def flow_matvec_sub(
    inp: Tensor,
    flow: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Subtract the output of [`flow_matvec`](index.html#jitfields.regularisers.flow_matvec)
    from `inp`.
    """
    return flow_matvec_add(inp, flow, weight,
                            absolute, membrane, bending, shears, div,
                            bound, voxel_size, out, True)


def flow_matvec_sub_(
    inp: Tensor,
    flow: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
) -> Tensor:
    """
    Subtract the output of [`flow_matvec`](index.html#jitfields.regularisers.flow_matvec)
    from `inp` (inplace).
    """
    return flow_matvec_add_(inp, flow, weight,
                             absolute, membrane, bending, shears, div,
                             bound, voxel_size, True)


def flow_kernel(
    shape: OneOrSeveral[int],
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
    **backend,
) -> Tensor:
    """Return the kernel of a Toeplitz regularization matrix.

    Parameters
    ----------
    shape : `int or list[int]`
        Number of spatial dimensions or shape of the tensor.
    absolute : float
        Penalty on absolute values.
    membrane : float
        Penalty on first derivatives.
    bending : float
        Penalty on second derivatives.
    shears : float
        Penalty on local shears. Linear elastic energy's `mu`.
    div : float
        Penalty on local volume changes. Linear elastic energy's `lambda`.
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    out : `(*shape, ndim, [ndim]) tensor`, optional
        Output placeholder, with shape `(*shape, ndim, [ndim])`.

    Returns
    -------
    out : `(*shape, ndim, [ndim]) tensor`
        Convolution kernel.
        A matrix or kernels with shape `(*shape, ndim, ndim)`
        if `shears` or `div`, else a vector of kernels with shape
        `(*shape, ndim)`.
    """
    if isinstance(shape, int):
        ndim = shape
        shape = [5 if bending else 3] * ndim
    else:
        ndim = len(shape)
        shape = list(shape)
    shape += [ndim]
    if shears or div:
        shape += [ndim]

    # allocate output
    if out is None:
        out = torch.empty(shape, **backend)

    bound = ensure_list(bound, out.shape[-1])
    voxel_size = ensure_list(voxel_size, out.shape[-1])

    # forward
    impl = cuda_impl if out.is_cuda else cpu_impl
    impl.flow_kernel(out, bound, voxel_size,
                     absolute, membrane, bending, shears, div)

    return out


def flow_kernel_add(
    inp: Tensor,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
    _sub: bool = False,
) -> Tensor:
    """
    Add the output of [`flow_kernel`](index.html#jitfields.regularisers.flow_kernel)
    to `inp`.
    """
    # allocate output
    if out is None:
        out = inp.clone()
    else:
        out.copy_(inp)

    bound = ensure_list(bound, inp.shape[-1])
    voxel_size = ensure_list(voxel_size, inp.shape[-1])

    # forward
    impl = cuda_impl if out.is_cuda else cpu_impl
    impl.flow_kernel(out, bound, voxel_size,
                     absolute, membrane, bending, shears, div,
                     'sub' if _sub else 'add')

    return out


def flow_kernel_add_(
    inp: Tensor,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    _sub: bool = False,
) -> Tensor:
    """
    Add the output of [`flow_kernel`](index.html#jitfields.regularisers.flow_kernel)
    to `inp` (inplace).
    """
    bound = ensure_list(bound, inp.shape[-1])
    voxel_size = ensure_list(voxel_size, inp.shape[-1])

    # forward
    impl = cuda_impl if inp.is_cuda else cpu_impl
    impl.flow_kernel(inp, bound, voxel_size,
                     absolute, membrane, bending, shears, div,
                     'sub' if _sub else 'add')
    return inp


def flow_kernel_sub(
    inp: Tensor,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Subtract the output of [`flow_kernel`](index.html#jitfields.regularisers.flow_kernel)
    from `inp`.
    """
    return flow_kernel_add(inp, absolute, membrane, bending, shears, div,
                           bound, voxel_size, out)


def flow_kernel_sub_(
    inp: Tensor,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
) -> Tensor:
    """
    Subtract the output of [`flow_kernel`](index.html#jitfields.regularisers.flow_kernel)
    from `inp` (inplace).
    """
    return flow_kernel_add_(inp, absolute, membrane, bending, shears, div,
                            bound, voxel_size)


def flow_diag(
    shape: OneOrSeveral[int],
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
    **backend,
) -> Tensor:
    """Return the diagonal of a regularization matrix.

    Parameters
    ----------
    shape : `list[int]`
        Shape of the tensor.
    weight : `(*batch, *spatial) tensor`, optional
        Weight map, to spatially modulate the regularization.
        With shape `(*batch, *spatial)`.
    absolute : `float`
        Penalty on absolute values.
    membrane : `float`
        Penalty on first derivatives.
    bending : `float`
        Penalty on second derivatives.
    shears : `float`
        Penalty on local shears.
    div : `float`
        Penalty on local volume changes.
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    out : `(*batch, *spatial, ndim) tensor`, optional
        Output placeholder

    Returns
    -------
    out : `(*batch, *spatial, ndim) tensor`
        Diagonal of the regularisation matrix,
        with shape `(*batch, *spatial, ndim)`.
    """
    ndim = len(shape)

    if weight is not None:
        batch = weight.shape[:-ndim]
        backend = dict(dtype=weight.dtype, device=weight.device)
        weight = [weight]
    else:
        batch = []
        weight = []

    # allocate output
    if out is None:
        out = torch.empty([*batch, *shape, len(shape)], **backend)

    # forward
    impl = cuda_impl if out.is_cuda else cpu_impl
    fn = impl.flow_diag_rls if weight else impl.flow_diag

    bound = ensure_list(bound, out.shape[-1])
    voxel_size = ensure_list(voxel_size, out.shape[-1])

    fn(out, *weight, bound, voxel_size,
       absolute, membrane, bending, shears, div)

    return out


def flow_diag_add(
    inp: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
    _sub: bool = False,
) -> Tensor:
    """
    Add the output of [`flow_diag`](index.html#jitfields.regularisers.flow_diag)
    to `inp`.
    """
    impl = cuda_impl if inp.is_cuda else cpu_impl

    if weight is not None:
        fn = impl.flow_diag_rls
        inp, weight = broadcast(inp, weight[..., None], skip_last=1)
        weight = weight[..., 0]
        weight = [weight]
    else:
        fn = impl.flow_diag
        weight = []

    # allocate output
    if out is None:
        out = inp.clone()
    else:
        out.copy_(inp)

    bound = ensure_list(bound, out.shape[-1])
    voxel_size = ensure_list(voxel_size, out.shape[-1])

    # forward
    fn(out, *weight, bound, voxel_size,
       absolute, membrane, bending, shears, div,
       'sub' if _sub else 'add')

    return out


def flow_diag_add_(
    inp: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    _sub: bool = False,
) -> Tensor:
    """
    Add the output of [`flow_diag`](index.html#jitfields.regularisers.flow_diag)
    to `inp` (inplace).
    """
    impl = cuda_impl if inp.is_cuda else cpu_impl

    if weight is not None:
        fn = impl.flow_diag_rls
        inp, weight = broadcast(inp, weight[..., None], skip_last=1)
        weight = weight[..., 0]
        weight = [weight]
    else:
        fn = impl.flow_diag
        weight = []

    bound = ensure_list(bound, inp.shape[-1])
    voxel_size = ensure_list(voxel_size, inp.shape[-1])

    # forward
    fn(inp, *weight, bound, voxel_size,
       absolute, membrane, bending, shears, div,
       'sub' if _sub else 'add')

    return inp


def flow_diag_sub(
    inp: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Subtract the output of [`flow_diag`](index.html#jitfields.regularisers.flow_diag)
    from `inp`.
    """
    return flow_diag_add(inp, weight,
                         absolute, membrane, bending, shears, div,
                         bound, voxel_size, out, True)


def flow_diag_sub_(
    inp: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
) -> Tensor:
    """
    Subtract the output of [`flow_diag`](index.html#jitfields.regularisers.flow_diag)
    from `inp` (inplace).
    """
    return flow_diag_add_(inp, weight,
                          absolute, membrane, bending, shears, div,
                          bound, voxel_size, True)


def flow_precond(
    mat: Tensor,
    vec: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Apply the preconditioning `(M + diag(R)) \ v`

    Parameters
    ----------
    mat : `(*batch, *spatial, DD) tensor`
        Preconditioning matrix `M` with shape `(*batch, *spatial, DD)`,
        where `DD` is one of `{1, D, D*(D+1)//2, D*D}`.
    vec : `(*batch, *spatial, D) tensor`
        Point `v` at which to solve the system,
        with shape `(*batch, *spatial, D)`.
    weight : `(`*batch, *spatial) tensor`, optional
        Regularization weight map, with shape `(*batch, *spatial)`.
    absolute : `float`
        Penalty on absolute values.
    membrane : `float`
        Penalty on first derivatives.
    bending : `float`
        Penalty on second derivatives.
    shears : `float`
        Penalty on local shears.
    div : `float`
        Penalty on local volume changes.
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    out : `(*batch, *spatial, D) tensor`
        Output placeholder, with shape

    Returns
    -------
    out : `(*batch, *spatial, D) tensor`
        Preconditioned vector.

    """
    ndim = vec.shape[-1]
    shape = vec.shape[-ndim-1:-1]
    diag = flow_diag(shape, weight,
                     absolute, membrane, bending, shears, div,
                     bound, voxel_size)
    return sym_solve(mat, vec, diag, out=out)


def flow_precond_(
    mat: Tensor,
    vec: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
) -> Tensor:
    r"""
    Apply the preconditioning `(M + diag(R)) \ v` inplace.
    See `flow_precond`.
    """
    ndim = vec.shape[-1]
    shape = vec.shape[-ndim-1:-1]
    diag = flow_diag(shape, weight,
                     absolute, membrane, bending, shears, div,
                     bound, voxel_size)
    return sym_solve_(mat, vec, diag)


def flow_forward(
    mat: Tensor,
    vec: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Apply the forward matrix-vector product `(M + R) @ v`

    Parameters
    ----------
    mat : `(*batch, *spatial, DD) tensor`
        Block-diagonal matrix with shape `(*batch, *spatial, DD)`, where
        `DD` is one of `{1, D, D*(D+1)//2, D*D}`.
    vec : `(*batch, *spatial, D) tensor`
        Point `v` at which to solve the system,
        with shape `(*batch, *spatial, D)`
    weight : `(*batch, *spatial) tensor`, optional
        Regularization weight map, with shape `(*batch, *spatial)`.
    absolute : `float`
        Penalty on absolute values.
    membrane : `float`
        Penalty on first derivatives.
    bending : `float`
        Penalty on second derivatives.
    shears : `float`
        Penalty on local shears.
    div : `float`
        Penalty on local volume changes.
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    out : `(*batch, *spatial, D) tensor`
        Output placeholder, with shape `(*batch, *spatial, D)`.

    Returns
    -------
    out : `(*batch, *spatial, D) tensor`
        Preconditioned vector, with shape `(*batch, *spatial, D)`.

    """
    out = sym_matvec(mat, vec, out=out)
    out = flow_matvec_add_(out, vec, weight,
                           absolute, membrane, bending, shears, div,
                           bound, voxel_size)
    return out


def flow_relax_(
    flow: Tensor,
    hes: Tensor,
    grd: Tensor,
    weight: Optional[Tensor] = None,
    absolute: float = 0,
    membrane: float = 0,
    bending: float = 0,
    shears: float = 0,
    div: float = 0,
    bound: OneOrSeveral[BoundType] = 'dft',
    voxel_size: OneOrSeveral[float] = 1,
    nb_iter: int = 1,
) -> Tensor:
    """Perform relaxation iterations (inplace).

    Parameters
    ----------
    flow : `(*batch, *spatial, ndim) tensor`
        Warm start, with shape `(*batch, *spatial, ndim)`.
    hes : `(*batch, *spatial, ndim*(ndim+1)//2) tensor`
        Input symmetric Hessian, in voxels.
        With shape `(*batch, *spatial, ndim*(ndim+1)//2)`.
    grd : `(*batch, *spatial, ndim) tensor`
        Input gradient, in voxels.
        With shape `(*batch, *spatial, ndim)`.
    weight : `(*batch, *spatial) tensor`, optional
        Weight map, to spatially modulate the regularization.
        With shape `(*batch, *spatial)`.
    absolute : `float`
        Penalty on absolute values.
    membrane : `float`
        Penalty on first derivatives.
    bending : `float`
        Penalty on second derivatives.
    shears : `float`
        Penalty on local shears.
    div : `float`
        Penalty on local volume changes.
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        Boundary conditions.
    voxel_size : `[sequence of] float`
        Voxel size.
    nb_iter : `int`
        Number of iterations

    Returns
    -------
    flow : `(*batch, *spatial, ndim) tensor`
        Refined solution with shape `(*batch, *spatial, ndim)` .
    """
    impl = cuda_impl if flow.is_cuda else cpu_impl

    # broadcast
    hes, grd = broadcast(hes, grd, skip_last=1)
    flow, grd = broadcast(flow, grd, skip_last=1)
    hes, grd = broadcast(hes, grd, skip_last=1)
    if weight is not None:
        flow, weight = broadcast(flow, weight[..., None], skip_last=1)
        weight = weight[..., 0]
        fn = impl.flow_relax_rls_
        weight = [weight]
    else:
        fn = impl.flow_relax_
        weight = []

    bound = ensure_list(bound, flow.shape[-1])
    voxel_size = ensure_list(voxel_size, flow.shape[-1])

    # forward
    fn(flow, hes, grd, *weight, nb_iter, bound, voxel_size,
       absolute, membrane, bending, shears, div)

    return flow