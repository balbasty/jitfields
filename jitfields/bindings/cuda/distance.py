from .utils import (
    culaunch, to_cupy, get_cuda_num_threads, get_cuda_blocks,
    get_offset_type, CachedKernel)
from ..common.utils import cinfo, bound_as_cname, spline_as_cname, to_np_dtype
import cupy as cp
import torch
import math


kernels_l1 = CachedKernel('distance_l1.cu')
kernels_l2 = CachedKernel('distance_euclidean.cu')
kernels_spline = CachedKernel('distance_spline.cu')
kernels_mesh = CachedKernel('distance_mesh.cu')


def l1dt_1d_(f, dim=-1, w=1):
    """in-place one-dimensional L1 distance"""
    f = f.movedim(dim, -1)
    ndim = f.ndim
    n = f.shape[:-1].numel()

    cuf = to_cupy(f)

    scalar_t = cuf.dtype
    offset_t = get_offset_type(cuf)
    shape, stride = cinfo(cuf, dtype=offset_t, backend=cp)

    asscalar = (cp.float16 if scalar_t == cp.float16 else
                cp.float32 if scalar_t == cp.float32 else
                cp.float64)

    kernel = kernels_l1.get('kernel', ndim, scalar_t, offset_t)
    culaunch(kernel, n, (cuf, asscalar(w), shape, stride))
    return f.movedim(-1, dim)


def l1dt_1d(f, dim=-1, w=1):
    """out-of-place one-dimensional L1 distance"""
    dtype = f.dtype
    if not f.dtype.is_floating_point:
        f = torch.get_default_dtype()
    f = f.to(dtype, copy=True)
    return l1dt_1d_(f, dim, w)


def edt_1d_(f, dim=-1, w=1):
    """in-place one-dimensional Euclidean distance"""
    ndim = f.ndim
    f = f.movedim(dim, -1)
    n = f.shape[:-1].numel()

    cuf = to_cupy(f)

    scalar_t = cuf.dtype
    offset_t = get_offset_type(cuf)
    shape, stride = cinfo(cuf, dtype=offset_t, backend=cp)

    nb_blocks, nb_threads = get_cuda_blocks(n), get_cuda_num_threads()
    buf = nb_blocks * nb_threads * f.shape[-1]
    buf *= 2 * cuf.dtype.itemsize + stride.dtype.itemsize
    buf = cp.empty([buf], dtype=cp.uint8)

    asscalar = (cp.float16 if scalar_t == cp.float16 else
                cp.float32 if scalar_t == cp.float32 else
                cp.float64)

    kernel = kernels_l2.get('kernel', ndim, scalar_t, offset_t)
    culaunch(kernel, n, (cuf, buf, asscalar(w), shape, stride))
    return f.movedim(-1, dim)


def edt_1d(f, dim=-1, w=1):
    """out-of-place one-dimensional Euclidean distance"""
    dtype = f.dtype
    if not f.dtype.is_floating_point:
        f = torch.get_default_dtype()
    f = f.to(dtype, copy=True)
    return edt_1d_(f, dim, w)



def splinedt_table_(time, dist, loc, coeff, timetable, order, bound):
    """in-place distance to 1D spline"""
    ndim = coeff.shape[-1]
    batch = coeff.shape[:-2]
    nbatch = len(batch)
    ntimes = timetable.shape[-1]

    cutime = to_cupy(time)
    cudist = to_cupy(dist)
    culoc = to_cupy(loc)
    cucoeff = to_cupy(coeff)
    cutimetable = to_cupy(timetable)

    scalar_t = cucoeff.dtype
    offset_t = get_offset_type(cutime, cudist, culoc, cucoeff, cutimetable)
    shape, stride_coeff = cinfo(cucoeff, dtype=offset_t, backend=cp)
    _, stride_time = cinfo(cutime, dtype=offset_t, backend=cp)
    _, stride_dist = cinfo(cudist, dtype=offset_t, backend=cp)
    _, stride_loc = cinfo(culoc, dtype=offset_t, backend=cp)
    _, stride_timetable = cinfo(cutimetable, dtype=offset_t, backend=cp)
    order = spline_as_cname(order)
    bound = bound_as_cname(bound)

    kernel = kernels_spline.get('mindist_table', nbatch, ndim, order, bound, scalar_t, offset_t)
    culaunch(kernel, batch.numel(), 
             (cutime, cudist, culoc, cucoeff, cutimetable, offset_t(ntimes), shape, 
              stride_time, stride_dist, stride_loc, stride_coeff, stride_timetable))
    return time, dist


def splinedt_quad_(time, dist, loc, coeff, order, bound, max_iter, tol, step):
    """in-place distance to 1D spline"""
    ndim = coeff.shape[-1]
    batch = coeff.shape[:-2]
    nbatch = len(batch)

    cutime = to_cupy(time)
    cudist = to_cupy(dist)
    culoc = to_cupy(loc)
    cucoeff = to_cupy(coeff)

    scalar_t = to_np_dtype(cucoeff.dtype).type
    offset_t = get_offset_type(cutime, cudist, culoc, cucoeff)
    shape, stride_coeff = cinfo(cucoeff, dtype=offset_t, backend=cp)
    _, stride_time = cinfo(cutime, dtype=offset_t, backend=cp)
    _, stride_dist = cinfo(cudist, dtype=offset_t, backend=cp)
    _, stride_loc = cinfo(culoc, dtype=offset_t, backend=cp)
    order = spline_as_cname(order)
    bound = bound_as_cname(bound)
    max_iter = offset_t(max_iter)
    tol = scalar_t(tol)
    step = scalar_t(step)

    kernel = kernels_spline.get('mindist_brent', nbatch, ndim, order, bound, scalar_t, offset_t)
    culaunch(kernel, batch.numel(), 
             (cutime, cudist, culoc, cucoeff, shape, 
              stride_time, stride_dist, stride_loc, stride_coeff, max_iter, tol, step))
    return time, dist


def splinedt_gaussnewton_(time, dist, loc, coeff, order, bound, max_iter, tol):
    """in-place distance to 1D spline"""
    ndim = coeff.shape[-1]
    batch = coeff.shape[:-2]
    nbatch = len(batch)

    cutime = to_cupy(time)
    cudist = to_cupy(dist)
    culoc = to_cupy(loc)
    cucoeff = to_cupy(coeff)

    scalar_t = to_np_dtype(cucoeff.dtype).type
    offset_t = get_offset_type(cutime, cudist, culoc, cucoeff)
    shape, stride_coeff = cinfo(cucoeff, dtype=offset_t, backend=cp)
    _, stride_time = cinfo(cutime, dtype=offset_t, backend=cp)
    _, stride_dist = cinfo(cudist, dtype=offset_t, backend=cp)
    _, stride_loc = cinfo(culoc, dtype=offset_t, backend=cp)
    order = spline_as_cname(order)
    bound = bound_as_cname(bound)
    max_iter = offset_t(max_iter)
    tol = scalar_t(tol)

    kernel = kernels_spline.get('mindist_gaussnewton', nbatch, ndim, order, bound, scalar_t, offset_t)
    culaunch(kernel, batch.numel(), 
             (cutime, cudist, culoc, cucoeff, shape, 
              stride_time, stride_dist, stride_loc, stride_coeff, max_iter, tol))
    return time, dist



def mesh_sdt(dist, coord, vertices, faces, tree, normals_faces, normal_vertices, normals_edges):
    """Signed distance transform with binary tree search"""

    npdist = to_cupy(dist)
    npfaces = to_cupy(faces)
    npvertices = to_cupy(vertices)
    nptree = to_cupy(tree)
    npcoord = to_cupy(coord)
    npnormfaces = to_cupy(normals_faces)
    npnormvertices = to_cupy(normal_vertices)
    npnormedges = to_cupy(normals_edges) if normals_edges is not None else None
    scalar_t = cp.dtype(npvertices.dtype)
    index_t = cp.dtype(npfaces.dtype)
    offset_t = cp.dtype(get_offset_type(
        npdist, npfaces, npvertices, npcoord, npnormfaces, npnormvertices, npnormedges,
    ))

    nbatch = npdist.ndim
    N, D = vertices.shape
    M, K = faces.shape
    if (D != K) or (D not in (2, 3)):
        raise ValueError('Faces must be triangles (in 3D) or segments (in 2D)')
    if (coord.shape[-1] != D):
        raise ValueError('Number of spatial dimensions not consistent.')

    nb_levels = int(math.ceil(math.log2(M))) + 2
    nb_threads = get_cuda_num_threads()
    nb_blocks =  get_cuda_blocks(dist.numel(), nb_threads)
    nptree_trace = cp.zeros([nb_levels * nb_threads * nb_blocks], dtype='uint8')

    size, stride_dist = cinfo(npdist, dtype=offset_t, backend=cp)
    _, stride_coord = cinfo(npcoord, dtype=offset_t, backend=cp)
    _, stride_faces = cinfo(npfaces, dtype=offset_t, backend=cp)
    _, stride_vertices = cinfo(npvertices, dtype=offset_t, backend=cp)
    _, stride_normfaces = cinfo(npnormfaces, dtype=offset_t, backend=cp)
    _, stride_normvertices = cinfo(npnormvertices, dtype=offset_t, backend=cp)
    if normals_edges is not None:
        _, stride_normedges = cinfo(npnormedges, dtype=offset_t, backend=cp)
    else:
        stride_normedges = None
    M = offset_t.type(M)
    N = offset_t.type(N)
    nb_levels = offset_t.type(nb_levels)

    kernel = kernels_mesh.get('sdt', nbatch, D, scalar_t, index_t, offset_t)
    culaunch(kernel, dist.numel(),
             (npdist, npcoord, npvertices, npfaces, nptree, nptree_trace, nb_levels,
              npnormfaces, npnormvertices, npnormedges,
              size, stride_dist, stride_coord, stride_vertices, stride_faces, 
              stride_normfaces, stride_normvertices, stride_normedges))

    return dist


def mesh_sdt_naive(dist, coord, vertices, faces, normals_faces, normal_vertices, normals_edges):
    """Naive signed distance transform (exhaustive search)"""

    npdist = to_cupy(dist)
    npfaces = to_cupy(faces)
    npvertices = to_cupy(vertices)
    npcoord = to_cupy(coord)
    npnormfaces = to_cupy(normals_faces)
    npnormvertices = to_cupy(normal_vertices)
    npnormedges = to_cupy(normals_edges) if normals_edges is not None else None
    scalar_t = cp.dtype(npvertices.dtype)
    index_t = cp.dtype(npfaces.dtype)
    offset_t = cp.dtype(get_offset_type(
        npdist, npfaces, npvertices, npcoord, npnormfaces, npnormvertices, npnormedges,
    ))

    nbatch = npdist.ndim
    N, D = vertices.shape
    M, K = faces.shape
    if (D != K) or (D not in (2, 3)):
        raise ValueError('Faces must be triangles (in 3D) or segments (in 2D)')
    if (coord.shape[-1] != D):
        raise ValueError('Number of spatial dimensions not consistent.')

    size, stride_dist = cinfo(npdist, dtype=offset_t, backend=cp)
    _, stride_coord = cinfo(npcoord, dtype=offset_t, backend=cp)
    _, stride_faces = cinfo(npfaces, dtype=offset_t, backend=cp)
    _, stride_vertices = cinfo(npvertices, dtype=offset_t, backend=cp)
    _, stride_normfaces = cinfo(npnormfaces, dtype=offset_t, backend=cp)
    _, stride_normvertices = cinfo(npnormvertices, dtype=offset_t, backend=cp)
    if normals_edges is not None:
        _, stride_normedges = cinfo(npnormedges, dtype=offset_t, backend=cp)
    else:
        stride_normedges = None
    M = offset_t.type(M)
    N = offset_t.type(N)

    kernel = kernels_mesh.get('sdt_naive', nbatch, D, scalar_t, index_t, offset_t)
    culaunch(kernel, dist.numel(),
             (npdist, npcoord, npvertices, npfaces, 
              npnormfaces, npnormvertices, npnormedges,
              size, M, stride_dist, stride_coord, stride_vertices, stride_faces, 
              stride_normfaces, stride_normvertices, stride_normedges))

    return dist

def mesh_dt(dist, coord, vertices, faces, tree):
    """Unsigned distance transform with binary tree search"""

    npdist = to_cupy(dist)
    npfaces = to_cupy(faces)
    npvertices = to_cupy(vertices)
    nptree = to_cupy(tree)
    npcoord = to_cupy(coord)
    scalar_t = cp.dtype(npvertices.dtype)
    index_t = cp.dtype(npfaces.dtype)
    offset_t = cp.dtype(get_offset_type(
        npdist, npfaces, npvertices, npcoord,
    ))

    nbatch = npdist.ndim
    N, D = vertices.shape
    M, K = faces.shape
    if (D != K) or (D not in (2, 3)):
        raise ValueError('Faces must be triangles (in 3D) or segments (in 2D)')
    if (coord.shape[-1] != D):
        raise ValueError('Number of spatial dimensions not consistent.')

    size, stride_dist = cinfo(npdist, dtype=offset_t, backend=cp)
    _, stride_coord = cinfo(npcoord, dtype=offset_t, backend=cp)
    _, stride_faces = cinfo(npfaces, dtype=offset_t, backend=cp)
    _, stride_vertices = cinfo(npvertices, dtype=offset_t, backend=cp)
    M = offset_t.type(M)
    N = offset_t.type(N)

    kernel = kernels_mesh.get('dt', nbatch, D, scalar_t, index_t, offset_t)
    culaunch(kernel, dist.numel(),
             (npdist, npcoord, npvertices, npfaces, nptree, 
              size, stride_dist, stride_coord, stride_vertices, stride_faces))

    return dist


def mesh_dt_naive(dist, coord, vertices, faces):
    """Naive unsigned distance transform (exhaustive search)"""

    npdist = to_cupy(dist)
    npfaces = to_cupy(faces)
    npvertices = to_cupy(vertices)
    npcoord = to_cupy(coord)
    scalar_t = cp.dtype(npvertices.dtype)
    index_t = cp.dtype(npfaces.dtype)
    offset_t = cp.dtype(get_offset_type(
        npdist, npfaces, npvertices, npcoord,
    ))

    nbatch = npdist.ndim
    N, D = vertices.shape
    M, K = faces.shape
    if (D != K) or (D not in (2, 3)):
        raise ValueError('Faces must be triangles (in 3D) or segments (in 2D)')
    if (coord.shape[-1] != D):
        raise ValueError('Number of spatial dimensions not consistent.')

    size, stride_dist = cinfo(npdist, dtype=offset_t, backend=cp)
    _, stride_coord = cinfo(npcoord, dtype=offset_t, backend=cp)
    _, stride_faces = cinfo(npfaces, dtype=offset_t, backend=cp)
    _, stride_vertices = cinfo(npvertices, dtype=offset_t, backend=cp)
    M = offset_t.type(M)
    N = offset_t.type(N)

    kernel = kernels_mesh.get('dt_naive', nbatch, D, scalar_t, index_t, offset_t)
    culaunch(kernel, dist.numel(),
             (npdist, npcoord, npvertices, npfaces, 
              size, M, stride_dist, stride_coord, stride_vertices, stride_faces))

    return dist