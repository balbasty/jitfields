import cppyy
import torch
import math
import numpy as np
from .utils import include, cwrap, nullptr
from ..common.utils import ctypename, cinfo, spline_as_cname, bound_as_cname

# cppyy.set_debug(True)
include()
cppyy.include('distance_l1.hpp')
cppyy.include('distance_euclidean.hpp')
cppyy.include('distance_spline.hpp')
cppyy.include('distance_mesh.hpp')


def l1dt_1d_(f, dim=-1, w=1):
    """in-place one-dimensional L1 distance"""
    ndim = f.ndim
    npf = f.movedim(dim, -1).numpy()

    scalar_t = npf.dtype
    offset_t = np.int64
    shape, stride = cinfo(npf, dtype=offset_t)
    template = f'{ndim}, {ctypename(scalar_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_l1.loop[template])
    func(npf, float(w), shape, stride)

    return f


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
    npf = f.movedim(dim, -1).numpy()

    scalar_t = npf.dtype
    offset_t = np.int64
    shape, stride = cinfo(npf, dtype=offset_t)

    template = f'{ndim}, {ctypename(scalar_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_e.loop[template])
    func(npf, float(w), shape, stride)

    return f


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

    nptime = time.numpy()
    npdist = dist.numpy()
    nploc = loc.numpy()
    npcoeff = coeff.numpy()
    nptimetable = timetable.numpy()

    scalar_t = npcoeff.dtype
    offset_t = np.int64
    shape, stride_coeff = cinfo(npcoeff, dtype=offset_t)
    _, stride_time = cinfo(nptime, dtype=offset_t)
    _, stride_dist = cinfo(npdist, dtype=offset_t)
    _, stride_loc = cinfo(nploc, dtype=offset_t)
    _, stride_timetable = cinfo(nptimetable, dtype=offset_t)
    order = spline_as_cname(order)
    bound = bound_as_cname(bound)

    template = f'{nbatch}, {ndim}, {order}, {bound}, {ctypename(scalar_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_spline.mindist_table[template])
    func(nptime, npdist, nploc, npcoeff, nptimetable, int(ntimes), shape, 
         stride_time, stride_dist, stride_loc, stride_coeff, stride_timetable)

    return time, dist


def splinedt_quad_(time, dist, loc, coeff, order, bound, max_iter, tol, step):
    """in-place distance to 1D spline"""
    ndim = coeff.shape[-1]
    batch = coeff.shape[:-2]
    nbatch = len(batch)

    nptime = time.numpy()
    npdist = dist.numpy()
    nploc = loc.numpy()
    npcoeff = coeff.numpy()

    scalar_t = npcoeff.dtype
    offset_t = np.int64
    shape, stride_coeff = cinfo(npcoeff, dtype=offset_t)
    _, stride_time = cinfo(nptime, dtype=offset_t)
    _, stride_dist = cinfo(npdist, dtype=offset_t)
    _, stride_loc = cinfo(nploc, dtype=offset_t)
    order = spline_as_cname(order)
    bound = bound_as_cname(bound)
    max_iter = int(max_iter)
    tol = float(tol)
    step = float(step)

    template = f'{nbatch}, {ndim}, {order}, {bound}, {ctypename(scalar_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_spline.mindist_brent[template])
    func(nptime, npdist, nploc, npcoeff, shape, 
         stride_time, stride_dist, stride_loc, stride_coeff, max_iter, tol, step)
    return time, dist


def splinedt_gaussnewton_(time, dist, loc, coeff, order, bound, max_iter, tol):
    """in-place distance to 1D spline"""
    ndim = coeff.shape[-1]
    batch = coeff.shape[:-2]
    nbatch = len(batch)

    nptime = time.numpy()
    npdist = dist.numpy()
    nploc = loc.numpy()
    npcoeff = coeff.numpy()

    scalar_t = npcoeff.dtype
    offset_t = np.int64
    shape, stride_coeff = cinfo(npcoeff, dtype=offset_t)
    _, stride_time = cinfo(nptime, dtype=offset_t)
    _, stride_dist = cinfo(npdist, dtype=offset_t)
    _, stride_loc = cinfo(nploc, dtype=offset_t)
    order = spline_as_cname(order)
    bound = bound_as_cname(bound)
    max_iter = int(max_iter)
    tol = float(tol)

    template = f'{nbatch}, {ndim}, {order}, {bound}, {ctypename(scalar_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_spline.mindist_gaussnewton[template])
    func(nptime, npdist, nploc, npcoeff, shape, 
         stride_time, stride_dist, stride_loc, stride_coeff, max_iter, tol)
    return time, dist


def mesh_make_tree(vertices, faces):
    """
    Compute a binary search tree (modifies `faces` inplace)
    vertices : (N, D) tensor[float|double]
    faces : (M, K) tensor[int|long]
    tree : (log2(M) * F) tensor[uint8]
    """
    
    npfaces = faces.numpy()
    npvertices = vertices.numpy()
    scalar_t = np.dtype(npvertices.dtype)
    index_t = np.dtype(npfaces.dtype)
    offset_t = np.dtype(np.int64)

    N, D = vertices.shape
    M, K = faces.shape
    if (D != K) or (D not in (2, 3)):
        raise ValueError('Faces must be triangles (in 3D) or segments (in 2D)')
    nb_levels = int(math.ceil(math.log2(M))) + 3
    # I don't understand why I need `+3` and not just `+1`.
    # It is empirical to avoid segfaults...
    nb_nodes = int(math.ceil(sum(2**(i) for i in range(nb_levels))))
    nb_features = scalar_t.itemsize * 2*(D+1) + index_t.itemsize * 3

    tree = torch.empty(nb_nodes * nb_features, dtype=torch.uint8)
    nptree = tree.numpy()

    _, stride_faces = cinfo(npfaces, dtype=offset_t)
    _, stride_vertices = cinfo(npvertices, dtype=offset_t)
    M = offset_t.type(M)
    N = offset_t.type(N)

    template = f'{D}, {ctypename(scalar_t)}, {ctypename(index_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_mesh.build_tree[template])
    func(nptree, npfaces, npvertices, M, N, stride_faces, stride_vertices)

    return tree, faces


def mesh_pseudonormals(vertices, faces):
    """
    """
    N, D = vertices.shape
    M, K = faces.shape

    normals_faces = vertices.new_empty([M, D])
    if D == 3:
        normals_edges = vertices.new_empty([M, D, D])
    else:
        normals_edges = None
    normal_vertices = vertices.new_zeros([N, D]) # needs zero init to accumulate

    npfaces = faces.numpy()
    npvertices = vertices.numpy()
    npnormfaces = normals_faces.numpy()
    npnormvertices = normal_vertices.numpy()
    npnormedges = (normals_edges.numpy() if normals_edges is not None else 
                   nullptr(npnormfaces.dtype))
    scalar_t = np.dtype(npvertices.dtype)
    index_t = np.dtype(npfaces.dtype)
    offset_t = np.dtype(np.int64)

    _, stride_faces = cinfo(npfaces, dtype=offset_t)
    _, stride_vertices = cinfo(npvertices, dtype=offset_t)
    _, stride_normfaces = cinfo(npnormfaces, dtype=offset_t)
    _, stride_normvertices = cinfo(npnormvertices, dtype=offset_t)
    if normals_edges is not None:
        _, stride_normedges = cinfo(npnormedges, dtype=offset_t)
    else:
        stride_normedges = nullptr(offset_t)
    M = offset_t.type(M)
    N = offset_t.type(N)

    template = f'{D}, {ctypename(scalar_t)}, {ctypename(index_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_mesh.build_normals[template])
    func(npnormfaces, npnormvertices, npnormedges, npfaces, npvertices, M, N, 
         stride_normfaces, stride_normvertices, stride_normedges, stride_faces, stride_vertices)

    if D == 3:
        return normals_faces, normal_vertices, normals_edges
    else:
        return normals_faces, normal_vertices


def mesh_sdt(dist, nearest_vertex, coord, vertices, faces, tree, normals_faces, normal_vertices, normals_edges):
    """Signed distance transform with binary tree search"""

    npdist = dist.numpy()
    npfaces = faces.numpy()
    npnearest = (nearest_vertex.numpy() if nearest_vertex is not None else 
                 nullptr(npfaces.dtype))
    # npentity = (nearest_entity.numpy() if nearest_entity is not None else 
    #             nullptr(np.uint8))
    npvertices = vertices.numpy()
    nptree = tree.numpy()
    npcoord = coord.numpy()
    npnormfaces = normals_faces.numpy()
    npnormvertices = normal_vertices.numpy()
    npnormedges = (normals_edges.numpy() if normals_edges is not None else 
                   nullptr(npdist.dtype))
    scalar_t = np.dtype(npvertices.dtype)
    index_t = np.dtype(npfaces.dtype)
    offset_t = np.dtype(np.int64)

    N, D = vertices.shape
    M, K = faces.shape
    if (D != K) or (D not in (2, 3)):
        raise ValueError('Faces must be triangles (in 3D) or segments (in 2D)')
    if (coord.shape[-1] != D):
        raise ValueError('Number of spatial dimensions not consistent.')

    size, stride_dist = cinfo(npdist, dtype=offset_t)
    _, stride_coord = cinfo(npcoord, dtype=offset_t)
    _, stride_faces = cinfo(npfaces, dtype=offset_t)
    _, stride_vertices = cinfo(npvertices, dtype=offset_t)
    _, stride_normfaces = cinfo(npnormfaces, dtype=offset_t)
    _, stride_normvertices = cinfo(npnormvertices, dtype=offset_t)
    if normals_edges is not None:
        _, stride_normedges = cinfo(npnormedges, dtype=offset_t)
    else:
        stride_normedges = nullptr(offset_t)
    if nearest_vertex is not None:
        _, stride_nearest = cinfo(npnearest, dtype=offset_t)
    else:
        stride_nearest = nullptr(offset_t)
    # if nearest_entity is not None:
    #     _, stride_entity = cinfo(npentity, dtype=offset_t)
    # else:
    #     stride_entity = nullptr(offset_t)
    M = offset_t.type(M)
    N = offset_t.type(N)

    template = f'{npcoord.ndim-1}, {D}, {ctypename(scalar_t)}, {ctypename(index_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_mesh.sdt[template])
    func(npdist, npnearest, npcoord, npvertices, npfaces, nptree,
         npnormfaces, npnormvertices, npnormedges,
         size, stride_dist, stride_nearest, stride_coord, stride_vertices, stride_faces, 
         stride_normfaces, stride_normvertices, stride_normedges)

    return dist


def mesh_sdt_naive(dist, coord, vertices, faces, normals_faces, normal_vertices, normals_edges):
    """Naive signed distance transform (exhaustive search)"""

    npdist = dist.numpy()
    npfaces = faces.numpy()
    npvertices = vertices.numpy()
    npcoord = coord.numpy()
    npnormfaces = normals_faces.numpy()
    npnormvertices = normal_vertices.numpy()
    npnormedges = normals_edges.numpy() if normals_edges is not None else nullptr(npdist.dtype)
    scalar_t = np.dtype(npvertices.dtype)
    index_t = np.dtype(npfaces.dtype)
    offset_t = np.dtype(np.int64)

    N, D = vertices.shape
    M, K = faces.shape
    if (D != K) or (D not in (2, 3)):
        raise ValueError('Faces must be triangles (in 3D) or segments (in 2D)')
    if (coord.shape[-1] != D):
        raise ValueError('Number of spatial dimensions not consistent.')

    size, stride_dist = cinfo(npdist, dtype=offset_t)
    _, stride_coord = cinfo(npcoord, dtype=offset_t)
    _, stride_faces = cinfo(npfaces, dtype=offset_t)
    _, stride_vertices = cinfo(npvertices, dtype=offset_t)
    _, stride_normfaces = cinfo(npnormfaces, dtype=offset_t)
    _, stride_normvertices = cinfo(npnormvertices, dtype=offset_t)
    if normals_edges is not None:
        _, stride_normedges = cinfo(npnormedges, dtype=offset_t)
    else:
        stride_normedges = nullptr(offset_t)
    M = offset_t.type(M)
    N = offset_t.type(N)

    template = f'{npcoord.ndim-1}, {D}, {ctypename(scalar_t)}, {ctypename(index_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_mesh.sdt_naive[template])
    func(npdist, npcoord, npvertices, npfaces, 
         npnormfaces, npnormvertices, npnormedges,
         size, M, stride_dist, stride_coord, stride_vertices, stride_faces,
         stride_normfaces, stride_normvertices, stride_normedges)

    return dist

def mesh_dt(dist, coord, vertices, faces, tree):
    """Unsigned distance transform with binary tree search"""

    npdist = dist.numpy()
    npfaces = faces.numpy()
    npvertices = vertices.numpy()
    nptree = tree.numpy()
    npcoord = coord.numpy()
    scalar_t = np.dtype(npvertices.dtype)
    index_t = np.dtype(npfaces.dtype)
    offset_t = np.dtype(np.int64)

    N, D = vertices.shape
    M, K = faces.shape
    if (D != K) or (D not in (2, 3)):
        raise ValueError('Faces must be triangles (in 3D) or segments (in 2D)')
    if (coord.shape[-1] != D):
        raise ValueError('Number of spatial dimensions not consistent.')

    size, stride_dist = cinfo(npdist, dtype=offset_t)
    _, stride_coord = cinfo(npcoord, dtype=offset_t)
    _, stride_faces = cinfo(npfaces, dtype=offset_t)
    _, stride_vertices = cinfo(npvertices, dtype=offset_t)
    M = offset_t.type(M)
    N = offset_t.type(N)

    template = f'{npcoord.ndim-1}, {D}, {ctypename(scalar_t)}, {ctypename(index_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_mesh.dt[template])
    func(npdist, npcoord, npvertices, npfaces, nptree, 
         size, stride_dist, stride_coord, stride_vertices, stride_faces)

    return dist


def mesh_dt_naive(dist, coord, vertices, faces):
    """Naive unsigned distance transform (exhaustive search)"""

    npdist = dist.numpy()
    npfaces = faces.numpy()
    npvertices = vertices.numpy()
    npcoord = coord.numpy()
    scalar_t = np.dtype(npvertices.dtype)
    index_t = np.dtype(npfaces.dtype)
    offset_t = np.dtype(np.int64)

    N, D = vertices.shape
    M, K = faces.shape
    if (D != K) or (D not in (2, 3)):
        raise ValueError('Faces must be triangles (in 3D) or segments (in 2D)')
    if (coord.shape[-1] != D):
        raise ValueError('Number of spatial dimensions not consistent.')

    size, stride_dist = cinfo(npdist, dtype=offset_t)
    _, stride_coord = cinfo(npcoord, dtype=offset_t)
    _, stride_faces = cinfo(npfaces, dtype=offset_t)
    _, stride_vertices = cinfo(npvertices, dtype=offset_t)
    M = offset_t.type(M)
    N = offset_t.type(N)

    template = f'{npcoord.ndim-1}, {D}, {ctypename(scalar_t)}, {ctypename(index_t)}, {ctypename(offset_t)}'
    func = cwrap(cppyy.gbl.jf.distance_mesh.dt_naive[template])
    func(npdist, npcoord, npvertices, npfaces, 
         size, M, stride_dist, stride_coord, stride_vertices, stride_faces)

    return dist