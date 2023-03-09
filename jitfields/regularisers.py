__all__ = [
    'grid_vel2mom',
    'grid_vel2mom_add', 'grid_vel2mom_add_',
    'grid_vel2mom_sub', 'grid_vel2mom_sub_',
    'grid_kernel',
    'grid_kernel_add', 'grid_kernel_add_',
    'grid_kernel_sub', 'grid_kernel_sub_',
    'grid_diag',
    'grid_diag_add', 'grid_diag_add_',
    'grid_diag_sub', 'grid_diag_sub_',
    'grid_precond', 'grid_precond_',
    'grid_forward',
    'grid_relax_',

    'field_vel2mom',
    'field_vel2mom_add', 'field_vel2mom_add_',
    'field_vel2mom_sub', 'field_vel2mom_sub_',
    'field_kernel',
    'field_kernel_add', 'field_kernel_add_',
    'field_kernel_sub', 'field_kernel_sub_',
    'field_diag',
    'field_diag_add', 'field_diag_add_',
    'field_diag_sub', 'field_diag_sub_',
    'field_precond', 'field_precond_',
    'field_forward',
]

import torch
from .utils import try_import, ensure_list, broadcast
from .sym import sym_solve, sym_solve_, sym_matvec
cuda_impl = try_import('jitfields.bindings.cuda', 'regularisers')
cpu_impl = try_import('jitfields.bindings.cpp', 'regularisers')


def grid_vel2mom(vel, weight=None,
                 absolute=0, membrane=0, bending=0, shears=0, div=0,
                 bound='dft', voxel_size=1, out=None):
    """Apply a spatial regularization matrix.

    Parameters
    ----------
    vel : (*batch, *spatial, ndim) tensor
        Input displacement field, in voxels.
    weight : (*batch, *spatial) tensor, optional
        Weight map, to spatially modulate the regularization.
    absolute : float
        Penalty on absolute values.
    membrane : float
        Penalty on first derivatives.
    bending : float
        Penalty on second derivatives.
    shears : float
        Penalty on local shears.
    div : float
        Penalty on local volume changes.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
        Boundary conditions.
    voxel_size : [sequence of] float
        Voxel size.
    out : (*batch, *spatial, ndim) tensor, optional
        Output placeholder

    Returns
    -------
    out : (*batch, *spatial, ndim) tensor
    """
    impl = cuda_impl if vel.is_cuda else cpu_impl

    # broadcast
    if weight is not None:
        vel, weight = broadcast(vel, weight[..., None], skip_last=1)
        weight = weight[..., 0]
        fn = impl.grid_vel2mom_rls
        weight = [weight]
    else:
        fn = impl.grid_vel2mom
        weight = []

    # allocate output
    if out is None:
        out = torch.empty_like(vel)

    bound = ensure_list(bound, vel.shape[-1])
    voxel_size = ensure_list(voxel_size, vel.shape[-1])

    # forward
    fn(out, vel, *weight, bound, voxel_size,
       absolute, membrane, bending, shears, div)

    return out


def grid_vel2mom_add(inp, vel, weight=None,
                     absolute=0, membrane=0, bending=0, shears=0, div=0,
                     bound='dft', voxel_size=1, out=None, _sub=False):
    """See `grid_vel2mom`"""
    impl = cuda_impl if vel.is_cuda else cpu_impl

    # broadcast
    inp, vel = broadcast(inp, vel)
    if weight is not None:
        vel, weight = broadcast(vel, weight[..., None], skip_last=1)
        inp, vel = broadcast(inp, vel)
        weight = weight[..., 0]
        fn = impl.grid_vel2mom_rls
        weight = [weight]
    else:
        fn = impl.grid_vel2mom
        weight = []

    # allocate output
    if out is None:
        out = inp.clone()
    else:
        out.copy(inp)

    bound = ensure_list(bound, vel.shape[-1])
    voxel_size = ensure_list(voxel_size, vel.shape[-1])

    # forward
    fn(out, vel, *weight, bound, voxel_size,
       absolute, membrane, bending, shears, div,
       'sub' if _sub else 'add')

    return out


def grid_vel2mom_add_(inp, vel, weight=None,
                      absolute=0, membrane=0, bending=0, shears=0, div=0,
                      bound='dft', voxel_size=1, _sub=False):
    """See `grid_vel2mom`"""
    impl = cuda_impl if vel.is_cuda else cpu_impl

    # broadcast
    inp, vel = broadcast(inp, vel)
    if weight is not None:
        vel, weight = broadcast(vel, weight[..., None], skip_last=1)
        inp, vel = broadcast(inp, vel)
        weight = weight[..., 0]
        fn = impl.grid_vel2mom_rls
        weight = [weight]
    else:
        fn = impl.grid_vel2mom
        weight = []

    bound = ensure_list(bound, vel.shape[-1])
    voxel_size = ensure_list(voxel_size, vel.shape[-1])

    # forward
    fn(inp, vel, *weight, bound, voxel_size,
       absolute, membrane, bending, shears, div,
       'sub' if _sub else 'add')

    return inp


def grid_vel2mom_sub(inp, vel, weight=None,
                     absolute=0, membrane=0, bending=0, shears=0, div=0,
                     bound='dft', voxel_size=1, out=None):
    """See `grid_vel2mom`"""
    return grid_vel2mom_add(inp, vel, weight,
                            absolute, membrane, bending, shears, div,
                            bound, voxel_size, out, True)


def grid_vel2mom_sub_(inp, vel, weight=None,
                      absolute=0, membrane=0, bending=0, shears=0, div=0,
                      bound='dft', voxel_size=1):
    """See `grid_vel2mom`"""
    return grid_vel2mom_add_(inp, vel, weight,
                             absolute, membrane, bending, shears, div,
                             bound, voxel_size, True)


def grid_kernel(shape,
                absolute=0, membrane=0, bending=0, shears=0, div=0,
                bound='dft', voxel_size=1, out=None, **backend):
    """Return the kernel of a Toeplitz regularization matrix.

    Parameters
    ----------
    shape : int or list[int]
        Number of spatial dimensions or shape of the tensor
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
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
        Boundary conditions.
    voxel_size : [sequence of] float
        Voxel size.
    out : (*shape, ndim, [ndim]) tensor, optional
        Output placeholder

    Returns
    -------
    out : (*shape, ndim, [ndim]) tensor
        Convolution kernel.
        A matrix or kernels ([ndim, ndim]) if `shears` or `div`,
        else a vector of kernels ([ndim]) .
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
    impl.grid_kernel(out, bound, voxel_size,
                     absolute, membrane, bending, shears, div)

    return out


def grid_kernel_add(inp,
                    absolute=0, membrane=0, bending=0, shears=0, div=0,
                    bound='dft', voxel_size=1, out=None, _sub=False):
    """See `grid_kernel`"""
    # allocate output
    if out is None:
        out = inp.clone()
    else:
        out.copy_(inp)

    bound = ensure_list(bound, inp.shape[-1])
    voxel_size = ensure_list(voxel_size, inp.shape[-1])

    # forward
    impl = cuda_impl if out.is_cuda else cpu_impl
    impl.grid_kernel(out, bound, voxel_size,
                     absolute, membrane, bending, shears, div,
                     'sub' if _sub else 'add')

    return out


def grid_kernel_add_(inp,
                     absolute=0, membrane=0, bending=0, shears=0, div=0,
                     bound='dft', voxel_size=1, _sub=False):
    """See `grid_kernel`"""
    bound = ensure_list(bound, inp.shape[-1])
    voxel_size = ensure_list(voxel_size, inp.shape[-1])

    # forward
    impl = cuda_impl if inp.is_cuda else cpu_impl
    impl.grid_kernel(inp, bound, voxel_size,
                     absolute, membrane, bending, shears, div,
                     'sub' if _sub else 'add')
    return inp


def grid_kernel_sub(inp,
                    absolute=0, membrane=0, bending=0, shears=0, div=0,
                    bound='dft', voxel_size=1, out=None):
    """See `grid_kernel`"""
    return grid_kernel_add(inp, absolute, membrane, bending, shears, div,
                           bound, voxel_size, out)


def grid_kernel_sub_(inp,
                     absolute=0, membrane=0, bending=0, shears=0, div=0,
                     bound='dft', voxel_size=1):
    """See `grid_kernel`"""
    return grid_kernel_add_(inp, absolute, membrane, bending, shears, div,
                            bound, voxel_size)


def grid_diag(shape, weight=None,
              absolute=0, membrane=0, bending=0, shears=0, div=0,
              bound='dft', voxel_size=1, out=None, **backend):
    """Return the diagonal of a regularization matrix.

    Parameters
    ----------
    shape : list[int]
        Shape of the tensor
    weight : (*batch, *spatial) tensor, optional
        Weight map, to spatially modulate the regularization.
    absolute : float
        Penalty on absolute values.
    membrane : float
        Penalty on first derivatives.
    bending : float
        Penalty on second derivatives.
    shears : float
        Penalty on local shears.
    div : float
        Penalty on local volume changes.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
        Boundary conditions.
    voxel_size : [sequence of] float
        Voxel size.
    out : (*batch, *spatial, ndim) tensor, optional
        Output placeholder

    Returns
    -------
    out : (*batch, *spatial, ndim) tensor
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
    fn = impl.grid_diag_rls if weight else impl.grid_diag

    bound = ensure_list(bound, out.shape[-1])
    voxel_size = ensure_list(voxel_size, out.shape[-1])

    fn(out, *weight, bound, voxel_size,
       absolute, membrane, bending, shears, div)

    return out


def grid_diag_add(inp, weight=None,
                  absolute=0, membrane=0, bending=0, shears=0, div=0,
                  bound='dft', voxel_size=1, out=None, _sub=False):
    """See `grid_diag`"""
    impl = cuda_impl if inp.is_cuda else cpu_impl

    if weight is not None:
        fn = impl.grid_diag_rls
        inp, weight = broadcast(inp, weight[..., None], skip_last=1)
        weight = weight[..., 0]
        weight = [weight]
    else:
        fn = impl.grid_diag
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


def grid_diag_add_(inp, weight=None,
                   absolute=0, membrane=0, bending=0, shears=0, div=0,
                   bound='dft', voxel_size=1, _sub=False):
    """See `grid_diag`"""
    impl = cuda_impl if inp.is_cuda else cpu_impl

    if weight is not None:
        fn = impl.grid_diag_rls
        inp, weight = broadcast(inp, weight[..., None], skip_last=1)
        weight = weight[..., 0]
        weight = [weight]
    else:
        fn = impl.grid_diag
        weight = []

    bound = ensure_list(bound, inp.shape[-1])
    voxel_size = ensure_list(voxel_size, inp.shape[-1])

    # forward
    fn(inp, *weight, bound, voxel_size,
       absolute, membrane, bending, shears, div,
       'sub' if _sub else 'add')

    return inp


def grid_diag_sub(inp, weight=None,
                  absolute=0, membrane=0, bending=0, shears=0, div=0,
                  bound='dft', voxel_size=1, out=None):
    """See `grid_diag`"""
    return grid_diag_add(inp, weight,
                         absolute, membrane, bending, shears, div,
                         bound, voxel_size, out, True)


def grid_diag_sub_(inp, weight=None,
                   absolute=0, membrane=0, bending=0, shears=0, div=0,
                   bound='dft', voxel_size=1, _sub=False):
    """See `grid_diag`"""
    return grid_diag_add_(inp, weight,
                          absolute, membrane, bending, shears, div,
                          bound, voxel_size, True)


def grid_relax_(vel, hes, grd, weight=None,
                absolute=0, membrane=0, bending=0, shears=0, div=0,
                bound='dft', voxel_size=1, nb_iter=1):
    """Apply a spatial regularization matrix.

    Parameters
    ----------
    vel : (*batch, *spatial, ndim) tensor
        Input displacement field, in voxels.
    hes : (*batch, *spatial, ndim*(ndim+1)//2) tensor
        Input displacement field, in voxels.
    grd : (*batch, *spatial, ndim) tensor
        Input displacement field, in voxels.
    weight : (*batch, *spatial) tensor, optional
        Weight map, to spatially modulate the regularization.
    absolute : float
        Penalty on absolute values.
    membrane : float
        Penalty on first derivatives.
    bending : float
        Penalty on second derivatives.
    shears : float
        Penalty on local shears.
    div : float
        Penalty on local volume changes.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
        Boundary conditions.
    voxel_size : [sequence of] float
        Voxel size.
    nb_iter : int
        Number of iterations

    Returns
    -------
    vel : (*batch, *spatial, ndim) tensor
    """
    impl = cuda_impl if vel.is_cuda else cpu_impl

    # broadcast
    hes, grd = broadcast(hes, grd, skip_last=1)
    vel, grd = broadcast(vel, grd, skip_last=1)
    hes, grd = broadcast(hes, grd, skip_last=1)
    if weight is not None:
        vel, weight = broadcast(vel, weight[..., None], skip_last=1)
        weight = weight[..., 0]
        fn = impl.grid_relax_rls_
        weight = [weight]
    else:
        fn = impl.grid_relax_
        weight = []

    bound = ensure_list(bound, vel.shape[-1])
    voxel_size = ensure_list(voxel_size, vel.shape[-1])

    # forward
    fn(vel, hes, grd, *weight, nb_iter, bound, voxel_size,
       absolute, membrane, bending, shears, div)

    return vel


def grid_precond(mat, vec, weight=None,
                 absolute=0, membrane=0, bending=0, shears=0, div=0,
                 bound='dft', voxel_size=1, out=None):
    """Apply the preconditioning `(M + diag(R)) \ v`

    Parameters
    ----------
    mat : (*batch, *spatial, DD) tensor
        DD == 1 | D | D*(D+1)//2 | D*D
        Preconditioning matrix `M`
    vec : (*batch, *spatial, D) tensor
        Point `v` at which to solve the system.
    weight : (*batch, *spatial) tensor, optional
        Regularization weight map.
    absolute : float
        Penalty on absolute values.
    membrane : float
        Penalty on first derivatives.
    bending : float
        Penalty on second derivatives.
    shears : float
        Penalty on local shears.
    div : float
        Penalty on local volume changes.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
        Boundary conditions.
    voxel_size : [sequence of] float
        Voxel size.
    out : (*batch, *spatial, D) tensor
        Output placeholder.

    Returns
    -------
    out : (*batch, *spatial, D) tensor
        Preconditioned vector.

    """
    ndim = vec.shape[-1]
    shape = vec.shape[-ndim-1:-1]
    diag = grid_diag(shape, weight,
                     absolute, membrane, bending, shears, div,
                     bound, voxel_size)
    return sym_solve(mat, vec, diag, out=out)


def grid_precond_(mat, vec, weight=None,
                  absolute=0, membrane=0, bending=0, shears=0, div=0,
                  bound='dft', voxel_size=1):
    """See `grid_precond`"""
    ndim = vec.shape[-1]
    shape = vec.shape[-ndim-1:-1]
    diag = grid_diag(shape, weight,
                     absolute, membrane, bending, shears, div,
                     bound, voxel_size)
    return sym_solve_(mat, vec, diag)


def grid_forward(mat, vec, weight=None,
                 absolute=0, membrane=0, bending=0, shears=0, div=0,
                 bound='dft', voxel_size=1, out=None):
    """Apply the forward matrix-vector product `(M + R) @ v`

    Parameters
    ----------
    mat : (*batch, *spatial, DD) tensor
        DD == 1 | D | D*(D+1)//2 | D*D
    vec : (*batch, *spatial, D) tensor
        Point `v` at which to solve the system.
    weight : (*batch, *spatial) tensor, optional
        Regularization weight map.
    absolute : float
        Penalty on absolute values.
    membrane : float
        Penalty on first derivatives.
    bending : float
        Penalty on second derivatives.
    shears : float
        Penalty on local shears.
    div : float
        Penalty on local volume changes.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
        Boundary conditions.
    voxel_size : [sequence of] float
        Voxel size.
    out : (*batch, *spatial, D) tensor
        Output placeholder.

    Returns
    -------
    out : (*batch, *spatial, D) tensor
        Preconditioned vector.

    """
    out = sym_matvec(mat, vec, out=out)
    out = grid_vel2mom_add_(out, vec, weight,
                            absolute, membrane, bending, shears, div,
                            bound, voxel_size)
    return out


def field_vel2mom(ndim, vec, weight=None,
                  absolute=0, membrane=0, bending=0,
                  bound='dft', voxel_size=1, out=None):
    """Apply a spatial regularization matrix.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    vec : (*batch, *spatial, nc) tensor
        Input vector field.
    weight : (*batch, *spatial, nc|1) tensor, optional
        Weight map, to spatially modulate the regularization.
    absolute : float
        Penalty on absolute values.
    membrane : float
        Penalty on first derivatives.
    bending : float
        Penalty on second derivatives.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
        Boundary conditions.
    voxel_size : [sequence of] float
        Voxel size.
    out : (*batch, *spatial, nc) tensor, optional
        Output placeholder

    Returns
    -------
    out : (*batch, *spatial, nc) tensor
    """
    impl = cuda_impl if vec.is_cuda else cpu_impl

    # broadcast
    if weight is not None:
        vec, weight = broadcast(vec, weight[..., None], skip_last=1)
        weight = weight[..., 0]
        fn = impl.field_vel2mom_rls
        weight = [weight]
    else:
        fn = impl.field_vel2mom
        weight = []

    # allocate output
    if out is None:
        out = torch.empty_like(vec)

    nc = vec.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = ensure_list(voxel_size, ndim)
    absolute = ensure_list(absolute, nc)
    membrane = ensure_list(membrane, nc)
    bending = ensure_list(bending, nc)

    # forward
    fn(out, vec, *weight, bound, voxel_size,
       absolute, membrane, bending)

    return out


def field_vel2mom_add(ndim, inp, vec, weight=None,
                      absolute=0, membrane=0, bending=0,
                      bound='dft', voxel_size=1, out=None, _sub=False):
    """See `field_vel2mom`"""
    impl = cuda_impl if vec.is_cuda else cpu_impl

    # broadcast
    inp, vec = broadcast(inp, vec)
    if weight is not None:
        vec, weight = broadcast(vec, weight[..., None], skip_last=1)
        inp, vec = broadcast(inp, vec)
        weight = weight[..., 0]
        fn = impl.field_vel2mom_rls
        weight = [weight]
    else:
        fn = impl.field_vel2mom
        weight = []

    # allocate output
    if out is None:
        out = inp.clone()
    else:
        out.copy(inp)

    nc = vec.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = ensure_list(voxel_size, ndim)
    absolute = ensure_list(absolute, nc)
    membrane = ensure_list(membrane, nc)
    bending = ensure_list(bending, nc)

    # forward
    fn(out, vec, *weight, bound, voxel_size,
       absolute, membrane, bending,
       'sub' if _sub else 'add')

    return out


def field_vel2mom_add_(ndim, inp, vec, weight=None,
                       absolute=0, membrane=0, bending=0,
                       bound='dft', voxel_size=1, _sub=False):
    """See `field_vel2mom`"""
    impl = cuda_impl if vec.is_cuda else cpu_impl

    # broadcast
    inp, vec = broadcast(inp, vec)
    if weight is not None:
        vec, weight = broadcast(vec, weight[..., None], skip_last=1)
        inp, vec = broadcast(inp, vec)
        weight = weight[..., 0]
        fn = impl.field_vel2mom_rls
        weight = [weight]
    else:
        fn = impl.field_vel2mom
        weight = []

    nc = vec.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = ensure_list(voxel_size, ndim)
    absolute = ensure_list(absolute, nc)
    membrane = ensure_list(membrane, nc)
    bending = ensure_list(bending, nc)

    # forward
    fn(inp, vec, *weight, bound, voxel_size,
       absolute, membrane, bending,
       'sub' if _sub else 'add')

    return inp


def field_vel2mom_sub(ndim, inp, vel, weight=None,
                      absolute=0, membrane=0, bending=0,
                      bound='dft', voxel_size=1, out=None):
    """See `field_vel2mom`"""
    return field_vel2mom_add(ndim, inp, vel, weight,
                             absolute, membrane, bending,
                             bound, voxel_size, out, True)


def field_vel2mom_sub_(ndim, inp, vel, weight=None,
                       absolute=0, membrane=0, bending=0,
                       bound='dft', voxel_size=1):
    """See `field_vel2mom`"""
    return field_vel2mom_add_(ndim, inp, vel, weight,
                              absolute, membrane, bending,
                              bound, voxel_size, True)


def field_kernel(shape,
                 absolute=0, membrane=0, bending=0,
                 bound='dft', voxel_size=1, out=None, **backend):
    """Return the kernel of a Toeplitz regularization matrix.

    Parameters
    ----------
    shape : int or list[int]
        Number of spatial dimensions or shape of the tensor
    absolute : float
        Penalty on absolute values.
    membrane : float
        Penalty on first derivatives.
    bending : float
        Penalty on second derivatives.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
        Boundary conditions.
    voxel_size : [sequence of] float
        Voxel size.
    out : (*shape, nc) tensor, optional
        Output placeholder

    Returns
    -------
    out : (*shape, nc) tensor
        Convolution kernel.
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
    voxel_size = ensure_list(voxel_size, ndim)
    absolute = ensure_list(absolute, nc)
    membrane = ensure_list(membrane, nc)
    bending = ensure_list(bending, nc)

    # forward
    impl = cuda_impl if out.is_cuda else cpu_impl
    impl.field_kernel(out, bound, voxel_size,
                      absolute, membrane, bending)

    return out


def field_kernel_add(ndim, inp,
                     absolute=0, membrane=0, bending=0,
                     bound='dft', voxel_size=1, out=None, _sub=False):
    """See `grid_kernel`"""
    # allocate output
    if out is None:
        out = inp.clone()
    else:
        out.copy_(inp)

    nc = inp.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = ensure_list(voxel_size, ndim)
    absolute = ensure_list(absolute, nc)
    membrane = ensure_list(membrane, nc)
    bending = ensure_list(bending, nc)

    # forward
    impl = cuda_impl if out.is_cuda else cpu_impl
    impl.grid_kernel(out, bound, voxel_size,
                     absolute, membrane, bending,
                     'sub' if _sub else 'add')

    return out


def field_kernel_add_(ndim, inp,
                      absolute=0, membrane=0, bending=0,
                      bound='dft', voxel_size=1, _sub=False):
    """See `grid_kernel`"""
    nc = inp.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = ensure_list(voxel_size, ndim)
    absolute = ensure_list(absolute, nc)
    membrane = ensure_list(membrane, nc)
    bending = ensure_list(bending, nc)

    # forward
    impl = cuda_impl if inp.is_cuda else cpu_impl
    impl.grid_kernel(inp, bound, voxel_size,
                     absolute, membrane, bending,
                     'sub' if _sub else 'add')
    return inp


def field_kernel_sub(ndim, inp,
                     absolute=0, membrane=0, bending=0,
                     bound='dft', voxel_size=1, out=None):
    """See `field_kernel`"""
    return field_kernel_add(ndim, inp, absolute, membrane, bending,
                            bound, voxel_size, out)


def field_kernel_sub_(ndim, inp,
                      absolute=0, membrane=0, bending=0,
                      bound='dft', voxel_size=1):
    """See `field_kernel`"""
    return field_kernel_add_(ndim, inp, absolute, membrane, bending,
                             bound, voxel_size)


def field_diag(shape, weight=None,
               absolute=0, membrane=0, bending=0,
               bound='dft', voxel_size=1, out=None, **backend):
    """Return the diagonal of a regularization matrix.

    Parameters
    ----------
    shape : list[int]
        Shape of the tensor
    weight : (*batch, *spatial, nc|1) tensor, optional
        Weight map, to spatially modulate the regularization.
    absolute : float
        Penalty on absolute values.
    membrane : float
        Penalty on first derivatives.
    bending : float
        Penalty on second derivatives.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
        Boundary conditions.
    voxel_size : [sequence of] float
        Voxel size.
    out : (*batch, *spatial, nc) tensor, optional
        Output placeholder

    Returns
    -------
    out : (*batch, *spatial, nc) tensor
    """
    ndim = len(shape)

    if weight is not None:
        batch = weight.shape[:-ndim]
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
        out = torch.empty([*batch, *shape, nc], **backend)

    # forward
    impl = cuda_impl if out.is_cuda else cpu_impl
    fn = impl.field_diag_rls if weight else impl.field_diag

    bound = ensure_list(bound, ndim)
    voxel_size = ensure_list(voxel_size, ndim)
    absolute = ensure_list(absolute, nc)
    membrane = ensure_list(membrane, nc)
    bending = ensure_list(bending, nc)

    fn(out, *weight, bound, voxel_size,
       absolute, membrane, bending)

    return out


def field_diag_add(ndim, inp, weight=None,
                   absolute=0, membrane=0, bending=0,
                   bound='dft', voxel_size=1, out=None, _sub=False):
    """See `grid_diag`"""
    impl = cuda_impl if inp.is_cuda else cpu_impl

    if weight is not None:
        fn = impl.field_diag_rls
        inp, weight = broadcast(inp, weight[..., None], skip_last=1)
        weight = weight[..., 0]
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
    voxel_size = ensure_list(voxel_size, ndim)
    absolute = ensure_list(absolute, nc)
    membrane = ensure_list(membrane, nc)
    bending = ensure_list(bending, nc)

    # forward
    fn(out, *weight, bound, voxel_size,
       absolute, membrane, bending,
       'sub' if _sub else 'add')

    return out


def field_diag_add_(ndim, inp, weight=None,
                    absolute=0, membrane=0, bending=0,
                    bound='dft', voxel_size=1, _sub=False):
    """See `grid_diag`"""
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
    voxel_size = ensure_list(voxel_size, ndim)
    absolute = ensure_list(absolute, nc)
    membrane = ensure_list(membrane, nc)
    bending = ensure_list(bending, nc)

    # forward
    fn(inp, *weight, bound, voxel_size,
       absolute, membrane, bending,
       'sub' if _sub else 'add')

    return inp


def field_diag_sub(inp, weight=None,
                   absolute=0, membrane=0, bending=0,
                   bound='dft', voxel_size=1, out=None):
    """See `field_diag`"""
    return field_diag_add(inp, weight,
                          absolute, membrane, bending,
                          bound, voxel_size, out, True)


def field_diag_sub_(inp, weight=None,
                    absolute=0, membrane=0, bending=0,
                    bound='dft', voxel_size=1, _sub=False):
    """See `field_diag`"""
    return field_diag_add_(inp, weight,
                           absolute, membrane, bending,
                           bound, voxel_size, True)


def field_precond(ndim, mat, vec, weight=None,
                  absolute=0, membrane=0, bending=0,
                  bound='dft', voxel_size=1, out=None):
    """Apply the preconditioning `(M + diag(R)) \ v`

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    mat : (*batch, *spatial, CC) tensor
        CC == 1 | C | C*(C+1)//2 | C*C
        Preconditioning matrix `M`
    vec : (*batch, *spatial, C) tensor
        Point `v` at which to solve the system.
    weight : (*batch, *spatial, nc|1) tensor, optional
        Regularization weight map.
    absolute : float
        Penalty on absolute values.
    membrane : float
        Penalty on first derivatives.
    bending : float
        Penalty on second derivatives.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
        Boundary conditions.
    voxel_size : [sequence of] float
        Voxel size.
    out : (*batch, *spatial, C) tensor
        Output placeholder.

    Returns
    -------
    out : (*batch, *spatial, C) tensor
        Preconditioned vector.

    """
    nc = vec.shape[-1]
    shape = vec.shape[-ndim-1:-1]
    absolute = ensure_list(absolute, nc)
    membrane = ensure_list(membrane, nc)
    bending = ensure_list(bending, nc)
    diag = field_diag(shape, weight,
                      absolute, membrane, bending,
                      bound, voxel_size)
    return sym_solve(mat, vec, diag, out=out)


def field_precond_(ndim, mat, vec, weight=None,
                   absolute=0, membrane=0, bending=0,
                   bound='dft', voxel_size=1):
    """See `field_precond`"""
    nc = vec.shape[-1]
    shape = vec.shape[-ndim-1:-1]
    absolute = ensure_list(absolute, nc)
    membrane = ensure_list(membrane, nc)
    bending = ensure_list(bending, nc)
    diag = field_diag(shape, weight,
                      absolute, membrane, bending,
                      bound, voxel_size)
    return sym_solve_(mat, vec, diag)


def field_forward(ndim, mat, vec, weight=None,
                  absolute=0, membrane=0, bending=0,
                  bound='dft', voxel_size=1, out=None):
    """Apply the forward matrix-vector product `(M + R) @ v`

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    mat : (*batch, *spatial, CC) tensor
        CC == 1 | C | C*(C+1)//2 | C*C
    vec : (*batch, *spatial, D) tensor
        Point `v` at which to solve the system.
    weight : (*batch, *spatial, nc|1) tensor, optional
        Regularization weight map.
    absolute : float
        Penalty on absolute values.
    membrane : float
        Penalty on first derivatives.
    bending : float
        Penalty on second derivatives.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
        Boundary conditions.
    voxel_size : [sequence of] float
        Voxel size.
    out : (*batch, *spatial, C) tensor
        Output placeholder.

    Returns
    -------
    out : (*batch, *spatial, C) tensor
        Preconditioned vector.

    """
    out = sym_matvec(mat, vec, out=out)
    out = field_vel2mom_add_(ndim, out, vec, weight,
                             absolute, membrane, bending,
                             bound, voxel_size)
    return out
