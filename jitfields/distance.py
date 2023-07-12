__all__ = [
    'euclidean_distance_transform',
    'l1_distance_transform',
    'signed_distance_transform',
    'spline_distance_table',
    'spline_distance_brent', 'spline_distance_brent_',
    'spline_distance_gaussnewton', 'spline_distance_gaussnewton_',
    'mesh_distance', 'mesh_distance_signed',
]

import torch
from torch import Tensor as tensor
from typing import Optional, Union, Tuple
from .utils import make_vector, try_import
from .typing import OneOrSeveral, BoundType, OrderType

cuda_dist = try_import('jitfields.bindings.cuda', 'distance')
cpu_dist = try_import('jitfields.bindings.cpp', 'distance')


def euclidean_distance_transform(
    x: tensor,
    ndim: Optional[int] = None,
    vx: OneOrSeveral[float] = 1,
    dtype: Optional[torch.dtype] = None,
) -> tensor:
    """Compute the Euclidean distance transform of a binary image

    Parameters
    ----------
    x : `(..., *spatial) tensor`
        Input tensor, with shape `(..., *spatial)`.
    ndim : `int`, default=`x.ndim`
        Number of spatial dimensions. Default: all.
    vx : `[sequence of] float`, default=1
        Voxel size.
    dtype : `torch.dtype`, optional
        Ouptut data type. Default is same as `x` if it has a floating
        point data type, else `torch.get_default_dtype()`.

    Returns
    -------
    d : `(..., *spatial) tensor`
        Distance map, with shape `(..., *spatial)`.

    References
    ----------
    1. Felzenszwalb, P.F. and Huttenlocher, D.P., 2012.
    [*Distance transforms of sampled functions.*](https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf)
    _Theory of computing_, 8(1), pp.415-428.

            @article{felzenszwalb2012distance,
              title={Distance transforms of sampled functions},
              author={Felzenszwalb, Pedro F and Huttenlocher, Daniel P},
              journal={Theory of computing},
              volume={8},
              number={1},
              pages={415--428},
              year={2012},
              publisher={Theory of Computing Exchange},
              url={https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf}
            }
    """
    if x.is_cuda:
        edt_1d_ = cuda_dist.edt_1d_
        l1dt_1d_ = cuda_dist.l1dt_1d_
    else:
        edt_1d_ = cpu_dist.edt_1d_
        l1dt_1d_ = cpu_dist.l1dt_1d_
    dtype = dtype or x.dtype
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    x = x.to(dtype, copy=True)
    x.masked_fill_(x > 0, float('inf'))
    ndim = ndim or x.ndim
    vx = make_vector(vx, ndim, dtype=torch.float).tolist()
    x = l1dt_1d_(x, -ndim, vx[0])
    if ndim > 1:
        x.square_()
        for d, w in zip(range(1, ndim), vx[1:]):
            x = edt_1d_(x, d-ndim, w)
        x.sqrt_()
    return x


def l1_distance_transform(
    x: tensor,
    ndim: Optional[int] = None,
    vx: OneOrSeveral[float] = 1,
    dtype: Optional[torch.dtype] = None,
) -> tensor:
    """Compute the L1 distance transform of a binary image

    Parameters
    ----------
    x : `(..., *spatial) tensor`
        Input tensor, with shape `(..., *spatial)`.
    ndim : `int`, default=`x.ndim`
        Number of spatial dimensions. Default: all.
    vx : `[sequence of] float`, default=1
        Voxel size.
    dtype : `torch.dtype`, optional
        Ouptut data type. Default is same as `x` if it has a floating
        point data type, else `torch.get_default_dtype()`.

    Returns
    -------
    d : `(..., *spatial) tensor`
        Distance map, with shape `(..., *spatial)`.

    References
    ----------
    1. Felzenszwalb, P.F. and Huttenlocher, D.P., 2012.
    [*Distance transforms of sampled functions.*](https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf)
    _Theory of computing_, 8(1), pp.415-428.

            @article{felzenszwalb2012distance,
              title={Distance transforms of sampled functions},
              author={Felzenszwalb, Pedro F and Huttenlocher, Daniel P},
              journal={Theory of computing},
              volume={8},
              number={1},
              pages={415--428},
              year={2012},
              publisher={Theory of Computing Exchange},
              url={https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf}
            }
    """
    if x.is_cuda:
        l1dt_1d_ = cuda_dist.l1dt_1d_
    else:
        l1dt_1d_ = cpu_dist.l1dt_1d_
    dtype = dtype or x.dtype
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    x = x.to(dtype, copy=True)
    x.masked_fill_(x > 0, float('inf'))
    ndim = ndim or x.ndim
    vx = make_vector(vx, ndim, dtype=torch.float).tolist()
    for d, w in enumerate(vx):
        x = l1dt_1d_(x, d-ndim, w)
    return x


def signed_distance_transform(
    x: tensor,
    ndim: Optional[int] = None,
    vx: OneOrSeveral[float] = 1,
    dtype: Optional[torch.dtype] = None,
) -> tensor:
    """Compute the Euclidean distance transform of a binary image

    Parameters
    ----------
    x : `(..., *spatial) tensor`
        Input tensor, with shape `(..., *spatial)`.
    ndim : `int`, default=`x.ndim`
        Number of spatial dimensions. Default: all.
    vx : `[sequence of] float`, default=1
        Voxel size.
    dtype : `torch.dtype`, optional
        Ouptut data type. Default is same as `x` if it has a floating
        point data type, else `torch.get_default_dtype()`.

    Returns
    -------
    d : `(..., *spatial) tensor`
        Distance map, with shape `(..., *spatial)`.

    References
    ----------
    1. Felzenszwalb, P.F. and Huttenlocher, D.P., 2012.
    [*Distance transforms of sampled functions.*](https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf)
    _Theory of computing_, 8(1), pp.415-428.

            @article{felzenszwalb2012distance,
              title={Distance transforms of sampled functions},
              author={Felzenszwalb, Pedro F and Huttenlocher, Daniel P},
              journal={Theory of computing},
              volume={8},
              number={1},
              pages={415--428},
              year={2012},
              publisher={Theory of Computing Exchange},
              url={https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf}
            }
    """
    x = x > 0
    d = euclidean_distance_transform(x, ndim, vx, dtype)
    d -= euclidean_distance_transform(x.logical_not_(), ndim, vx, dtype)
    return d



def _dot(x, y):
    """Dot product along the last dimension"""
    return x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1).squeeze(-1)


def spline_distance_table(
    loc: tensor, 
    coeff: tensor, 
    steps: Optional[Union[int, tensor]] = None, 
    order: OrderType = 3, 
    bound: BoundType = 'dct2', 
    square: bool = False,
) -> Tuple[tensor, tensor]:
    """Compute the minimum distance from a set of points to a 1D spline

    Parameters
    ----------
    loc : `(..., D) tensor`
        Point set.
    coeff : `(..., N, D) tensor`
        Spline coefficients encoding the location of the 1D spline.
    steps : `int or (..., K) tensor`
        Number of time steps to try, or list of time steps to try.
    order : {1..7}
        Spline order.
    bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`
        Boundary conditions of the spline.
    square : bool
        Return the squared Euclidean distance.

    Returns
    -------
    dist : `(...) tensor`
        Distance from each point in the set to its closest point on the spline
    time : `(...) tensor`
        Time of the closest point on the spline
    """

    fn = cuda_dist.splinedt_table_ if coeff.is_cuda else cpu_dist.splinedt_table_

    if steps is None:
        length = coeff[..., 1:, :] - coeff[..., :-1, :]
        length = _dot(length, length).sqrt_().sum(-1).max()
        steps = max(3, (length / 2).ceil().int().item())
    if isinstance(steps, int):
        steps = torch.linspace(0, coeff.shape[-2] - 1, steps, dtype=coeff.dtype, device=coeff.device)
    else:
        steps = torch.as_tensor(steps, dtype=coeff.dtype, device=coeff.device).flatten()

    batch = torch.broadcast_shapes(loc.shape[:-1], coeff.shape[:-2], steps.shape[:-1])
    loc = loc.expand(torch.Size(batch) + loc.shape[-1:])
    coeff = coeff.expand(torch.Size(batch) + coeff.shape[-2:])
    steps = steps.expand(torch.Size(batch) + steps.shape[-1:])
    time = loc.new_zeros(batch)
    dist = loc.new_full(batch, float('inf'))

    fn(time, dist, loc, coeff, steps, order, bound)
    if not square:
        dist = dist.sqrt_()

    return dist, time


def spline_distance_brent_(
    dist: tensor, 
    time: tensor, 
    loc: tensor, 
    coeff: tensor, 
    max_iter: int = 128, 
    tol: float = 1e-6, 
    step_size: float = 0.01, 
    order: OrderType = 3, 
    bound: BoundType = 'dct2', 
    square: bool = False,
) -> Tuple[tensor, tensor]:
    """Compute the minimum distance from a set of points to a 1D spline (inplace)

    Parameters
    ----------
    dist : `(...) tensor`
        Initial distance from each point in the set to its closest point on the spline
    time : `(...) tensor`
        Initial time of the closest point on the spline
    loc : `(..., D) tensor`
        Point set.
    coeff : `(..., N, D) tensor`
        Spline coefficients encoding the location of the 1D spline.
    max_iter : int
        Number of optimization steps.
    tol : float
        Tolerance for early stopping
    step_size : float
        Initial search size.
    order : {1..7}
        Spline order.
    bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`
        Boundary conditions of the spline.
    square : bool
        Return the squared Euclidean distance.

    Returns
    -------
    dist : `(...) tensor`
        Distance from each point in the set to its closest point on the spline
    time : `(...) tensor`
        Time of the closest point on the spline
    """

    fn = cuda_dist.splinedt_quad_ if coeff.is_cuda else cpu_dist.splinedt_quad_

    batch = torch.broadcast_shapes(dist.shape, time.shape, loc.shape[:-1], coeff.shape[:-2])
    loc = loc.expand(torch.Size(batch) + loc.shape[-1:])
    coeff = coeff.expand(torch.Size(batch) + coeff.shape[-2:])
    time = time.expand(batch)
    dist = dist.expand(batch)

    dist = dist.square_()
    fn(time, dist, loc, coeff, order, bound, max_iter, tol, step_size)
    if not square:
        dist = dist.sqrt_()

    return dist, time



def spline_distance_brent(
    loc: tensor, 
    coeff: tensor, 
    max_iter: int = 128, 
    tol: float = 1e-6, 
    step_size: float = 0.01, 
    order: OrderType = 3, 
    bound: BoundType = 'dct2', 
    square: bool = False,
    steps: Optional[Union[int, tensor]] = None, 
) -> Tuple[tensor, tensor]:
    """Compute the minimum distance from a set of points to a 1D spline

    Parameters
    ----------
    loc : `(..., D) tensor`
        Point set.
    coeff : `(..., N, D) tensor`
        Spline coefficients encoding the location of the 1D spline.
    max_iter : int
        Number of optimization steps.
    tol : float
        Tolerance for early stopping
    step_size : float
        Initial search size.
    order : {1..7}
        Spline order.
    bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`
        Boundary conditions of the spline.
    square : bool
        Return the squared Euclidean distance.
    steps : int
        Number of steps used in the table-based initialisation.

    Returns
    -------
    dist : `(...) tensor`
        Distance from each point in the set to its closest point on the spline
    time : `(...) tensor`
        Time of the closest point on the spline
    """

    dist, time = spline_distance_table(loc, coeff, order=order, bound=bound, steps=steps)
    return spline_distance_brent_(dist, time, loc, coeff, max_iter, tol, step_size, order, bound, square)


def spline_distance_gaussnewton_(
    dist: tensor, 
    time: tensor, 
    loc: tensor, 
    coeff: tensor, 
    max_iter: int = 16, 
    tol: float = 1e-6, 
    order: OrderType = 3, 
    bound: BoundType = 'dct2', 
    square: bool = False,
) -> Tuple[tensor, tensor]:
    """Compute the minimum distance from a set of points to a 1D spline (inplace)

    Parameters
    ----------
    dist : `(...) tensor`
        Initial distance from each point in the set to its closest point on the spline
    time : `(...) tensor`
        Initial time of the closest point on the spline
    loc : `(..., D) tensor`
        Point set.
    coeff : `(..., N, D) tensor`
        Spline coefficients encoding the location of the 1D spline.
    max_iter : int
        Number of optimization steps.
    tol : float
        Tolerance for early stopping
    order : {1..7}
        Spline order.
    bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`
        Boundary conditions of the spline.
    square : bool
        Return the squared Euclidean distance.

    Returns
    -------
    dist : `(...) tensor`
        Distance from each point in the set to its closest point on the spline
    time : `(...) tensor`
        Time of the closest point on the spline
    """

    fn = cuda_dist.splinedt_gaussnewton_ if coeff.is_cuda else cpu_dist.splinedt_gaussnewton_

    batch = torch.broadcast_shapes(dist.shape, time.shape, loc.shape[:-1], coeff.shape[:-2])
    loc = loc.expand(torch.Size(batch) + loc.shape[-1:])
    coeff = coeff.expand(torch.Size(batch) + coeff.shape[-2:])
    time = time.expand(batch)
    dist = dist.expand(batch)

    dist = dist.square_()
    fn(time, dist, loc, coeff, order, bound, max_iter, tol)
    if not square:
        dist = dist.sqrt_()

    return dist, time


def spline_distance_gaussnewton(
    loc: tensor, 
    coeff: tensor, 
    max_iter: int = 16, 
    tol: float = 1e-6, 
    order: OrderType = 3, 
    bound: BoundType = 'dct2', 
    square: bool = False,
    steps: Optional[Union[int, tensor]] = None, 
) -> Tuple[tensor, tensor]:
    """Compute the minimum distance from a set of points to a 1D spline

    Parameters
    ----------
    loc : `(..., D) tensor`
        Point set.
    coeff : `(..., N, D) tensor`
        Spline coefficients encoding the location of the 1D spline.
    max_iter : int
        Number of optimization steps.
    tol : float
        Tolerance for early stopping
    order : {1..7}
        Spline order.
    bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`
        Boundary conditions of the spline.
    square : bool
        Return the squared Euclidean distance.
    steps : int
        Number of steps used in the table-based initialisation.

    Returns
    -------
    dist : `(...) tensor`
        Distance from each point in the set to its closest point on the spline
    time : `(...) tensor`
        Time of the closest point on the spline
    """

    dist, time = spline_distance_table(loc, coeff, order=order, bound=bound, steps=steps)
    return spline_distance_gaussnewton_(dist, time, loc, coeff, max_iter, tol, order, bound, square)


def mesh_distance_signed(
    loc: tensor, 
    vertices: tensor, 
    faces: tensor,
    out: Optional[tensor] = None,
) -> tensor:
    """Compute the *signed* minimum distance from a set of points to a 1D spline

    Parameters
    ----------
    loc : `(..., D) tensor`
        Point set.
    vertices : `(N, D) tensor`
        Mesh vertices
    faces : `(M, D) tensor[integer]`
        Mesh faces

    Returns
    -------
    dist : `(...) tensor`
        Signed distance from each point in the set to its closest point on the mesh
        (negative inside, positive outside)
    """

    fn_sdt = cuda_dist.mesh_sdt if loc.is_cuda else cpu_dist.mesh_sdt
    fn_tree = cpu_dist.mesh_make_tree
    fn_norm = cpu_dist.mesh_pseudonormals

    # move to CPU (no choice for tree and normals)
    cpuvertices = vertices.cpu()
    cpufaces = faces.to('cpu', copy=True)

    # build binary search tree (modifies faces inplace)
    tree, cpufaces = fn_tree(cpuvertices, cpufaces)

    # compute pseudonormals
    if loc.shape[-1] == 3:
        normf, normv, norme = fn_norm(cpuvertices, cpufaces)
    else:
        normf, normv = fn_norm(cpuvertices, cpufaces)
        norme = None

    # move to loc's device
    # NOTE that faces were reordered to match tree order, so
    # we MUST transfer `cpufaces`, even if the input `faces` was
    # already on the gpu.
    vertices = vertices.to(loc)
    normf = normf.to(loc)
    normv = normv.to(loc)
    norme = norme.to(loc)
    faces = cpufaces.to(loc.device)
    tree = tree.to(loc.device)
    if out is None:
        out = loc.new_empty(loc.shape[:-1])

    # compute signed distance
    fn_sdt(out, loc, vertices, faces, tree, normf, normv, norme)

    return out


def mesh_distance(
    loc: tensor, 
    vertices: tensor, 
    faces: tensor,
    out: Optional[tensor] = None,
) -> tensor:
    """Compute the minimum distance from a set of points to a 1D spline

    Parameters
    ----------
    loc : `(..., D) tensor`
        Point set.
    vertices : `(N, D) tensor`
        Mesh vertices
    faces : `(M, D) tensor[integer]`
        Mesh faces

    Returns
    -------
    dist : `(...) tensor`
        Signed distance from each point in the set to its closest point on the mesh
        (negative inside, positive outside)
    """

    fn_dt = cuda_dist.mesh_dt if loc.is_cuda else cpu_dist.mesh_dt
    fn_tree = cpu_dist.mesh_make_tree

    # move to CPU (no choice for tree and normals)
    cpuvertices = vertices.cpu()
    cpufaces = faces.to('cpu', copy=True)

    # build binary search tree (modifies faces inplace)
    tree, cpufaces = fn_tree(cpuvertices, cpufaces)

    # move to loc's device
    # NOTE that faces were reordered to match tree order, so
    # we MUST transfer `cpufaces`, even if the input `faces` was
    # already on the gpu.
    vertices = vertices.to(loc)
    faces = cpufaces.to(loc.device)
    if out is None:
        out = loc.new_empty(out.shape[:-1])

    # compute unsigned distance
    fn_dt(out, loc, vertices, faces, tree)

    return out