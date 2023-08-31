#ifndef JF_DISTANCE_MESH_LOOP
#define JF_DISTANCE_MESH_LOOP
#include "../lib/atomic.h"
#include "../lib/cuda_switch.h"
#include "../lib/distance/mesh_cpp.h"
#include "../lib/batch.h"
#include "../lib/parallel.h"
#include <vector>

namespace jf {
namespace distance_mesh {


template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
void build_tree(
    uint8_t        * _tree,            // (log2(M) * F) tensor -> Placeholder for binary tree
    index_t        * _faces,           // (M, D) tensor -> All faces (face = D vertex indices)
    const scalar_t * _vertices,        // (N, D) tensor -> All vertices
    offset_t         nb_faces,         // M
    offset_t         nb_vertices,      // N
    const offset_t * stride_faces,     // [M, D] list -> Strides of `faces`
    const offset_t * stride_vertices   // [N, D] list -> Strides of `vertices`
)
{
    using Klass = MeshDistCpp<ndim, scalar_t, index_t, offset_t>;
    using Node = typename Klass::Node;
    using FaceList = StridedPointList<ndim, index_t, offset_t>;
    using VertexList = ConstStridedPointList<ndim, scalar_t, offset_t>;

    auto faces    = FaceList(_faces, stride_faces[0], stride_faces[1]);
    auto vertices = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto tree     = reinterpret_cast<Node *>(_tree);

    index_t node_id = 0;
    Klass::build_tree(tree, node_id, -1, 0, nb_faces, faces, vertices);
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
void build_normals(
    scalar_t       * _normfaces,            // (M, D) tensor
    scalar_t       * _normvertices,         // (N, D) tensor
    scalar_t       * _normedges,            // (M, D, D) tensor
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    offset_t         nb_faces,              // M
    offset_t         nb_vertices,           // N
    const offset_t * stride_normfaces,      // [M, D] list
    const offset_t * stride_normvertices,   // [N, D] list
    const offset_t * stride_normedges,      // [M, D, D] list
    const offset_t * stride_faces,          // [M, D] list -> Strides of `faces`
    const offset_t * stride_vertices        // [N, D] list -> Strides of `vertices`
)
{
    using Klass          = MeshDistUtilCpp<ndim, scalar_t, offset_t>;
    using FaceList       = ConstStridedPointListSized<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointListSized<ndim, scalar_t, offset_t>;
    using NormalList     = StridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = StridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;

    // If 2D -> no edges
    auto _stride_normedges = EdgeStride();
    if (stride_normedges)
        _stride_normedges.copy_(ConstRefPoint<3, offset_t>(stride_normedges));

    auto faces        = FaceList(_faces, stride_faces[0], stride_faces[1], nb_faces);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1], nb_vertices);
    auto normfaces    = NormalList(_normfaces, stride_normfaces[0], stride_normfaces[1]);
    auto normvertices = NormalList(_normvertices, stride_normvertices[0], stride_normvertices[1]);
    auto normedges    = EdgeNormalList(_normedges, _stride_normedges);

    Klass::build_normals(normfaces, normvertices, normedges, faces, vertices);
}

template <
    int nbatch,             // Number of batch dimensions in coord 
    int ndim,               // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
void sdt(
          scalar_t * dist,                  // (*batch) tensor -> Output placeholder for distance
          index_t  * nearest_vertex,        // (*batch) tensor -> Output placeholder for index of nearest vertex
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const uint8_t  * _tree,                 // (log2(M) * F) tensor -> Binary tree
    const scalar_t * _normfaces,            // (M, D) tensor
    const scalar_t * _normvertices,         // (N, D) tensor
    const scalar_t * _normedges,            // (M, D, D) tensor
    const offset_t * _size,                 // [*batch] list -> Size of `dist`
    const offset_t * _stride_dist,          // [*batch] list -> Strides of `dist`
    const offset_t * _stride_nearest,       // [*batch] list -> Strides of `nearest_vertex`
    const offset_t * _stride_coord,         // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces,          // [M, D] list -> Strides of `faces`
    const offset_t * stride_normfaces,      // [M, D] list
    const offset_t * stride_normvertices,   // [N, D] list
    const offset_t * stride_normedges       // [M, D, D] list
)
{
    using Klass          = MeshDistCpp<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using NormalList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = ConstStridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;

    StaticPoint<nbatch, offset_t> size;
    size.copy_(ConstRefPoint<nbatch, offset_t>(_size));
    StaticPoint<nbatch, offset_t> stride_dist;
    stride_dist.copy_(ConstRefPoint<nbatch, offset_t>(_stride_dist));
    StaticPoint<nbatch+1, offset_t> stride_coord;
    stride_coord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_coord));

    StaticPoint<nbatch, offset_t> stride_nearest;
    if (_stride_nearest)
        stride_nearest.copy_(ConstRefPoint<nbatch, offset_t>(_stride_nearest));

    // If 2D -> no edges
    auto _stride_normedges = EdgeStride();
    if (stride_normedges)
        _stride_normedges.copy_(ConstRefPoint<3, offset_t>(stride_normedges));

    auto faces        = FaceList(_faces, stride_faces[0], stride_faces[1]);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto normfaces    = NormalList(_normfaces, stride_normfaces[0], stride_normfaces[1]);
    auto normvertices = NormalList(_normvertices, stride_normvertices[0], stride_normvertices[1]);
    auto normedges    = EdgeNormalList(_normedges, _stride_normedges);
    auto tree         = reinterpret_cast<const Node *>(_tree);

    offset_t numel = size.prod();
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_coord = index2offset<nbatch>(i, size.data, stride_coord.data);
        offset_t offset_dist  = index2offset<nbatch>(i, size.data, stride_dist.data);
        offset_t offset_nearest = 0;
        if (nearest_vertex)
            offset_nearest  = index2offset<nbatch>(i, size.data, stride_nearest.data);

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::signed_sqdist(
            point,
            vertices, 
            faces, 
            tree,
            normfaces,
            normedges,
            normvertices,
            nearest_vertex + offset_nearest
        );
    }});
}

template <
    int      nbatch,        // Number of batch dimensions in coord 
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
void sdt_backward(
          scalar_t * gradcoord,             // (*batch, D) tensor -> Output placeholder for gradient wrt coord
          scalar_t * _gradvert,             // (N, D) tensor -> Output placeholder for gradient wrt vertices
    const scalar_t * grad,                  // (*batch) tensor -> Gradient wrt distance
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const uint8_t  * _tree,                 // (log2(M) * F) tensor -> Binary tree
    const scalar_t * _normfaces,            // (M, D) tensor
    const scalar_t * _normvertices,         // (N, D) tensor
    const scalar_t * _normedges,            // (M, D, D) tensor
    const offset_t * _size,                 // [*batch] list -> Size of `dist`
    const offset_t * _stride_gradcoord,     // [*batch] list -> Strides of `gradcoord`
    const offset_t * stride_gradvert,       // [N, D] list -> Strides of `gradvert`
    const offset_t * _stride_grad,          // [*batch] list -> Strides of `grad`
    const offset_t * _stride_coord,         // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces,          // [M, D] list -> Strides of `faces`
    const offset_t * stride_normfaces,      // [M, D] list
    const offset_t * stride_normvertices,   // [N, D] list
    const offset_t * _stride_normedges      // [M, D, D] list
)
{
    using Klass          = MeshDistCpp<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using GradVertexList = StridedPointList<ndim, scalar_t, offset_t>;
    using NormalList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = ConstStridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;

    StaticPoint<nbatch, offset_t> size;
    size.copy_(ConstRefPoint<nbatch, offset_t>(_size));
    StaticPoint<nbatch, offset_t> stride_grad;
    stride_grad.copy_(ConstRefPoint<nbatch, offset_t>(_stride_grad));
    StaticPoint<nbatch+1, offset_t> stride_coord;
    stride_coord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_coord));

    StaticPoint<nbatch+1, offset_t> stride_gradcoord;
    if (_stride_gradcoord)
        stride_gradcoord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_gradcoord));

    // If 2D -> no edges
    auto stride_normedges = EdgeStride();
    if (_stride_normedges)
        stride_normedges.copy_(ConstRefPoint<3, offset_t>(_stride_normedges));

    auto tree         = reinterpret_cast<const Node *>(_tree);
    auto faces        = FaceList(_faces, stride_faces[0], stride_faces[1]);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto normfaces    = NormalList(_normfaces, stride_normfaces[0], stride_normfaces[1]);
    auto normvertices = NormalList(_normvertices, stride_normvertices[0], stride_normvertices[1]);
    auto normedges    = EdgeNormalList(_normedges, stride_normedges);
    auto gradvert     = GradVertexList(_gradvert, 
                                       _gradvert ? stride_gradvert[0] : 0, 
                                       _gradvert ? stride_gradvert[1] : 0);

    // We need atomic add if computing derivatives wrt vertices, since
    // several faces may add into the same vertex. If not available, 
    // deactivate parallel loop.
    offset_t numel = size.prod();
    auto grain_size = (has_atomic_add<scalar_t>::value || !_gradvert) ? GRAIN_SIZE : numel;

    parallel_for(0, numel, grain_size, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        StaticPoint<ndim, scalar_t>             gradpoint;
        StaticPointList<ndim, ndim, scalar_t>   gradface;
        index_t                                 nearest_face;

        offset_t offset_grad  = index2offset<nbatch>(i, size.data, stride_grad.data);
        offset_t offset_coord = index2offset<nbatch>(i, size.data, stride_coord.data);
        auto grad1 = grad[offset_grad];

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(
                coord + offset_coord, stride_coord[nbatch]
        ));

        Klass::signed_sqdist_grad(
            nearest_face,
            gradpoint,
            gradface,
            point,
            vertices, 
            faces, 
            tree,
            normfaces,
            normedges,
            normvertices
        );

        if (gradcoord)
        {
            gradpoint.mul_(grad1);
            offset_t offset_gradcoord = index2offset<nbatch>(i, size.data, stride_gradcoord.data);
            StridedPoint<ndim, scalar_t, offset_t>(
                gradcoord + offset_gradcoord, stride_gradcoord[nbatch]
            ).copy_(gradpoint);
        }

        if (_gradvert)
        {
            auto face = faces[nearest_face];
            for (offset_t d=0; d<ndim; ++d)
            {
                auto vertex     = gradvert[face[d]];
                auto gradvertex = gradface[d];
                for (offset_t d=0; d<ndim; ++d)
                {
                    anyAtomicAddNoReturn(&(vertex[d]), gradvertex[d] * grad1);
                }
            }
        }
    }});
}


template <
    int      nbatch,        // Number of batch dimensions in coord 
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
void sdt_naive(
          scalar_t * dist,                  // (*batch) tensor -> Output placeholder for distance
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const scalar_t * _normfaces,            // (M, D) tensor
    const scalar_t * _normvertices,         // (N, D) tensor
    const scalar_t * _normedges,            // (M, D, D) tensor
    const offset_t * _size,                 // [*batch] list -> Size of `dist`
          offset_t   nb_faces,
    const offset_t * _stride_dist,          // [*batch] list -> Strides of `dist`
    const offset_t * _stride_coord,         // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces,          // [M, D] list -> Strides of `faces`
    const offset_t * stride_normfaces,      // [M, D] list
    const offset_t * stride_normvertices,   // [N, D] list
    const offset_t * stride_normedges       // [M, D, D] list
)
{
    using Klass          = MeshDistCpp<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointListSized<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using NormalList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = ConstStridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;


    StaticPoint<nbatch, offset_t> size;
    size.copy_(ConstRefPoint<nbatch, offset_t>(_size));
    StaticPoint<nbatch, offset_t> stride_dist;
    stride_dist.copy_(ConstRefPoint<nbatch, offset_t>(_stride_dist));
    StaticPoint<nbatch+1, offset_t> stride_coord;
    stride_coord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_coord));

    // In 2D -> no edges
    auto _stride_normedges = EdgeStride();
    if (stride_normedges)
        _stride_normedges.copy_(ConstRefPoint<3, offset_t>(stride_normedges));

    auto faces        = FaceList(_faces, stride_faces[0], stride_faces[1], nb_faces);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto normfaces    = NormalList(_normfaces, stride_normfaces[0], stride_normfaces[1]);
    auto normvertices = NormalList(_normvertices, stride_normvertices[0], stride_normvertices[1]);
    auto normedges    = EdgeNormalList(_normedges, _stride_normedges);

    offset_t numel = size.prod();
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_coord = index2offset<nbatch>(i, size.data, stride_coord.data);
        offset_t offset_dist  = index2offset<nbatch>(i, size.data, stride_dist.data);

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::signed_sqdist_naive(
            point,
            vertices, 
            faces, 
            normfaces,
            normedges,
            normvertices
        );
    }});
}

template <
    int      nbatch,        // Number of batch dimensions in coord 
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
void dt(
          scalar_t * dist,              // (*batch) tensor -> Output placeholder for distance
    const scalar_t * coord,             // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,         // (N, D) tensor -> All vertices
    const index_t  * _faces,            // (M, D) tensor -> All faces (face = D vertex indices)
    const uint8_t  * _tree,             // (log2(M) * F) tensor -> Binary tree
    const offset_t * _size,             // [*batch] list -> Size of `dist`
    const offset_t * _stride_dist,      // [*batch] list -> Strides of `dist`
    const offset_t * _stride_coord,     // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,   // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces       // [M, D] list -> Strides of `faces`
)
{
    using Klass          = MeshDistCpp<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;

    StaticPoint<nbatch, offset_t> size;
    size.copy_(ConstRefPoint<nbatch, offset_t>(_size));
    StaticPoint<nbatch, offset_t> stride_dist;
    stride_dist.copy_(ConstRefPoint<nbatch, offset_t>(_stride_dist));
    StaticPoint<nbatch+1, offset_t> stride_coord;
    stride_coord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_coord));

    auto faces    = FaceList(_faces, stride_faces[0], stride_faces[1]);
    auto vertices = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto tree     = reinterpret_cast<const Node *>(_tree);

    offset_t numel = size.prod();
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_coord = index2offset<nbatch>(i, size.data, stride_coord.data);
        offset_t offset_dist  = index2offset<nbatch>(i, size.data, stride_dist.data);

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::unsigned_sqdist(
            point,
            vertices, 
            faces, 
            tree
        );
    }});
}

template <
    int      nbatch,        // Number of batch dimensions in coord 
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
void dt_backward(
          scalar_t * gradcoord,             // (*batch, D) tensor -> Output placeholder for gradient wrt coord
          scalar_t * _gradvert,             // (N, D) tensor -> Output placeholder for gradient wrt vertices
    const scalar_t * grad,                  // (*batch) tensor -> Gradient wrt distance
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const uint8_t  * _tree,                 // (log2(M) * F) tensor -> Binary tree
    const offset_t * _size,                 // [*batch] list -> Size of `dist`
    const offset_t * _stride_gradcoord,     // [*batch] list -> Strides of `gradcoord`
    const offset_t * stride_gradvert,       // [N, D] list -> Strides of `gradvert`
    const offset_t * _stride_grad,          // [*batch] list -> Strides of `grad`
    const offset_t * _stride_coord,         // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces           // [M, D] list -> Strides of `faces`
)
{
    using Klass          = MeshDistCpp<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using GradVertexList = StridedPointList<ndim, scalar_t, offset_t>;

    StaticPoint<nbatch, offset_t> size;
    size.copy_(ConstRefPoint<nbatch, offset_t>(_size));
    StaticPoint<nbatch, offset_t> stride_grad;
    stride_grad.copy_(ConstRefPoint<nbatch, offset_t>(_stride_grad));
    StaticPoint<nbatch+1, offset_t> stride_coord;
    stride_coord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_coord));

    StaticPoint<nbatch+1, offset_t> stride_gradcoord;
    if (_stride_gradcoord)
        stride_gradcoord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_gradcoord));

    auto tree     = reinterpret_cast<const Node *>(_tree);
    auto faces    = FaceList(_faces, stride_faces[0], stride_faces[1]);
    auto vertices = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto gradvert = GradVertexList(_gradvert, 
                                   _gradvert ? stride_gradvert[0] : 0, 
                                   _gradvert ? stride_gradvert[1] : 0);

    // We need atomic add if computing derivatives wrt vertices, since
    // several faces may add into the same vertex. If not available, 
    // deactivate parallel loop.
    offset_t numel = size.prod();
    auto grain_size = (has_atomic_add<scalar_t>::value || !_gradvert) ? GRAIN_SIZE : numel;

    parallel_for(0, numel, grain_size, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        StaticPoint<ndim, scalar_t>             gradpoint;
        StaticPointList<ndim, ndim, scalar_t>   gradface;
        index_t                                 nearest_face;

        offset_t offset_coord = index2offset<nbatch>(i, size.data, stride_coord.data);
        offset_t offset_grad  = index2offset<nbatch>(i, size.data, stride_grad.data);
        auto grad1 = grad[offset_grad];

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(
                coord + offset_coord, stride_coord[nbatch]
        ));

        Klass::unsigned_sqdist_grad(
            nearest_face,
            gradpoint,
            gradface,
            point,
            vertices, 
            faces, 
            tree
        );

        if (gradcoord)
        {
            gradpoint.mul_(grad1);
            offset_t offset_gradcoord = index2offset<nbatch>(i, size.data, stride_gradcoord.data);
            StridedPoint<ndim, scalar_t, offset_t>(
                gradcoord + offset_gradcoord, stride_gradcoord[nbatch]
            ).copy_(gradpoint);
        }

        if (_gradvert)
        {
            auto face = faces[nearest_face];
            for (offset_t d=0; d<ndim; ++d)
            {
                auto vertex     = gradvert[face[d]];
                auto gradvertex = gradface[d];
                for (offset_t d=0; d<ndim; ++d)
                {
                    anyAtomicAddNoReturn(&(vertex[d]), gradvertex[d] * grad1);
                }
            }
        }
    }});
}

template <
    int      nbatch,        // Number of batch dimensions in coord 
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
void dt_naive(
          scalar_t * dist,              // (*batch) tensor -> Output placeholder for distance
    const scalar_t * coord,             // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,         // (N, D) tensor -> All vertices
    const index_t  * _faces,            // (M, D) tensor -> All faces (face = D vertex indices)
    const offset_t * _size,             // [*batch] list -> Size of `dist`
          offset_t   nb_faces,
    const offset_t * _stride_dist,      // [*batch] list -> Strides of `dist`
    const offset_t * _stride_coord,     // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,   // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces       // [M, D] list -> Strides of `faces`
)
{
    using Klass          = MeshDistCpp<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointListSized<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;

    StaticPoint<nbatch, offset_t> size;
    size.copy_(ConstRefPoint<nbatch, offset_t>(_size));
    StaticPoint<nbatch, offset_t> stride_dist;
    stride_dist.copy_(ConstRefPoint<nbatch, offset_t>(_stride_dist));
    StaticPoint<nbatch+1, offset_t> stride_coord;
    stride_coord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_coord));

    auto faces    = FaceList(_faces, stride_faces[0], stride_faces[1], nb_faces);
    auto vertices = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);

    offset_t numel = size.prod();
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_coord = index2offset<nbatch>(i, size.data, stride_coord.data);
        offset_t offset_dist  = index2offset<nbatch>(i, size.data, stride_dist.data);

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::unsigned_sqdist_naive(
            point,
            vertices, 
            faces
        );
    }});
}

} // namespace distance_spline
} // namespace jf

#endif // JF_DISTANCE_MESH_LOOP
