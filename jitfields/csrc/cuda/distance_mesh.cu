#include "../lib/cuda_switch.h"
#include "../lib/distance/mesh_cuda.h"
#include "../lib/batch.h"
#include "../lib/utils.h"

using namespace std;
using namespace jf;
using namespace jf::distance_mesh;

template <
    int nbatch,             // Number of batch dimensions in coord 
    int ndim,               // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
__global__ void sdt(
          scalar_t * dist,                  // (*batch) tensor -> Output placeholder for distance
          index_t  * nearest_vertex,        // (*batch) tensor -> Output placeholder for index of nearest vertex
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const char     * _tree,                 // (log2(M) * F) tensor -> Binary tree
          char     * _treetrace,            // (log2(M) * F) tensor -> Binary tree
          offset_t   treesize,
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
    const offset_t * _stride_normedges      // [M, D, D] list
)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    using Klass          = MeshDistCuda<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using NormalList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = ConstStridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;
    using RefPoint       = ConstStridedPoint<ndim, scalar_t, offset_t>;
    using ClonedPoint    = StaticPoint<ndim, scalar_t>;

    auto treetrace = SizedStridedPointer<char, offset_t>(
        _treetrace + index, stride, treesize
    );

    StaticPoint<nbatch, offset_t> size;
    size.copy_(ConstRefPoint<nbatch, offset_t>(_size));
    StaticPoint<nbatch, offset_t> stride_dist;
    stride_dist.copy_(ConstRefPoint<nbatch, offset_t>(_stride_dist));
    StaticPoint<nbatch+1, offset_t> stride_coord;
    stride_coord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_coord));

    StaticPoint<nbatch, offset_t> stride_nearest;
    if (_stride_nearest)
        stride_nearest.copy_(ConstRefPoint<nbatch, offset_t>(_stride_nearest));


    // In 2D -> no edges
    auto stride_normedges = EdgeStride();
    if (_stride_normedges)
        stride_normedges.copy_(ConstRefPoint<3, offset_t>(_stride_normedges));

    auto faces        = FaceList(_faces, stride_faces[0], stride_faces[1]);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto normfaces    = NormalList(_normfaces, stride_normfaces[0], stride_normfaces[1]);
    auto normvertices = NormalList(_normvertices, stride_normvertices[0], stride_normvertices[1]);
    auto normedges    = EdgeNormalList(_normedges, stride_normedges);
    auto tree         = reinterpret_cast<const Node *>(_tree);

    offset_t numel = size.prod();
    for (offset_t i=index; i < numel; i += stride)
    {
        if (i!=0) continue;

        for (offset_t j=0; j<treetrace.size; ++j)
            treetrace[j] = static_cast<char>(0);

        offset_t offset_coord = index2offset<nbatch>(i, size.data, stride_coord.data);
        offset_t offset_dist  = index2offset<nbatch>(i, size.data, stride_dist.data);
        offset_t offset_nearest = 0;
        if (nearest_vertex)
            offset_nearest  = index2offset<nbatch>(i, size.data, stride_nearest.data);

        ClonedPoint point;
        point.copy_(RefPoint(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::signed_sqdist(
            point,
            vertices, 
            faces, 
            tree,
            treetrace,
            normfaces,
            normedges,
            normvertices,
            nearest_vertex + offset_nearest
        );
        printf("%d %f\n", i, dist[offset_dist]);
    }
}

template <
    int nbatch,             // Number of batch dimensions in coord 
    int ndim,               // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
__global__ void sdt_backward(
          scalar_t * gradcoord,             // (*batch, D) tensor -> Output placeholder for gradient wrt coord
          scalar_t * _gradvert,             // (N, D) tensor -> Output placeholder for gradient wrt vertices
    const scalar_t * grad,                  // (*batch) tensor -> Gradient wrt distance
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const char     * _tree,                 // (log2(M) * F) tensor -> Binary tree
    char           * _treetrace,            // (log2(M) * F) tensor -> Binary tree
    offset_t         treesize,
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
    offset_t index  = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    using Klass          = MeshDistCuda<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using NormalList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = ConstStridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;
    using RefPoint       = ConstStridedPoint<ndim, scalar_t, offset_t>;
    using ClonedPoint    = StaticPoint<ndim, scalar_t>;

    auto treetrace = SizedStridedPointer<char, offset_t>(_treetrace + index, stride, treesize);

    StaticPoint<nbatch, offset_t> size;
    size.copy_(ConstRefPoint<nbatch, offset_t>(_size));
    StaticPoint<nbatch, offset_t> stride_grad;
    stride_grad.copy_(ConstRefPoint<nbatch, offset_t>(_stride_grad));
    StaticPoint<nbatch+1, offset_t> stride_coord;
    stride_coord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_coord));

    StaticPoint<nbatch+1, offset_t> stride_gradcoord;
    if (_stride_gradcoord)
        stride_gradcoord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_gradcoord));

    // In 2D -> no edges
    auto stride_normedges = EdgeStride();
    if (_stride_normedges)
        stride_normedges.copy_(ConstRefPoint<3, offset_t>(_stride_normedges));

    auto faces        = FaceList(_faces, stride_faces[0], stride_faces[1]);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto normfaces    = NormalList(_normfaces, stride_normfaces[0], stride_normfaces[1]);
    auto normvertices = NormalList(_normvertices, stride_normvertices[0], stride_normvertices[1]);
    auto normedges    = EdgeNormalList(_normedges, stride_normedges);
    auto gradvert     = GradVertexList(_gradvert, 
                                       _gradvert ? stride_gradvert[0] : 0, 
                                       _gradvert ? stride_gradvert[1] : 0);
    auto tree         = reinterpret_cast<const Node *>(_tree);

    offset_t numel = size.prod();
    for (offset_t i=index; i < numel; i += stride)
    {
        for (offset_t j=0; j<treetrace.size; ++j)
            treetrace[j] = static_cast<char>(0);

        StaticPoint<ndim, scalar_t>             gradpoint;
        StaticPointList<ndim, ndim, scalar_t>   gradface;
        index_t                                 nearest_face;

        offset_t offset_coord = index2offset<nbatch>(i, size.data, stride_coord.data);
        offset_t offset_grad  = index2offset<nbatch>(i, size.data, stride_grad.data);
        auto grad1 = grad[offset_grad];

        ClonedPoint point;
        point.copy_(RefPoint(coord + offset_coord, stride_coord[nbatch]));

        Klass::signed_sqdist_grad(
            nearest_face,
            gradpoint,
            gradface,
            point,
            vertices, 
            faces, 
            tree,
            treetrace,
            normfaces,
            normedges,
            normvertices
        );

        if (gradcoord)
        {
            gradpoint *= grad1;
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
    }
}

template <
    int nbatch,             // Number of batch dimensions in coord 
    int ndim,               // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
__global__ void sdt_naive(
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
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    using Klass          = MeshDistCuda<ndim, scalar_t, index_t, offset_t>;
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
    for (offset_t i=index; i < numel; i += stride)
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
    }
}

template <
    int nbatch,             // Number of batch dimensions in coord 
    int ndim,               // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
__global__ void dt(
          scalar_t * dist,                  // (*batch) tensor -> Output placeholder for distance
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const char     * _tree,                 // (log2(M) * F) tensor -> Binary tree
          char     * _treetrace,            // (log2(M) * F) tensor -> Binary tree
          offset_t   treesize,
    const offset_t * _size,                 // [*batch] list -> Size of `dist`
    const offset_t * _stride_dist,          // [*batch] list -> Strides of `dist`
    const offset_t * _stride_coord,         // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces           // [M, D] list -> Strides of `faces`
)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    using Klass          = MeshDistCuda<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;

    auto treetrace = SizedStridedPointer<char, offset_t>(_treetrace + index, stride, treesize);

    StaticPoint<nbatch, offset_t> size;
    size.copy_(ConstRefPoint<nbatch, offset_t>(_size));
    StaticPoint<nbatch, offset_t> stride_dist;
    stride_dist.copy_(ConstRefPoint<nbatch, offset_t>(_stride_dist));
    StaticPoint<nbatch+1, offset_t> stride_coord;
    stride_coord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_coord));

    auto faces        = FaceList(_faces, stride_faces[0], stride_faces[1]);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto tree         = reinterpret_cast<const Node *>(_tree);

    offset_t numel = size.prod();
    for (offset_t i=index; index < numel; index += stride, i=index)
    {
        for (offset_t j=0; j<treetrace.size; ++j)
            treetrace[j] = static_cast<char>(0);

        offset_t offset_coord = index2offset<nbatch>(i, size.data, stride_coord.data);
        offset_t offset_dist  = index2offset<nbatch>(i, size.data, stride_dist.data);

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::unsigned_sqdist(
            point,
            vertices, 
            faces, 
            tree,
            treetrace
        );
    }
}

template <
    int nbatch,             // Number of batch dimensions in coord 
    int ndim,               // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
__global__ void dt_backward(
          scalar_t * gradcoord,             // (*batch, D) tensor -> Output placeholder for gradient wrt coord
          scalar_t * _gradvert,             // (N, D) tensor -> Output placeholder for gradient wrt vertices
    const scalar_t * grad,                  // (*batch) tensor -> Gradient wrt distance
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const char     * _tree,                 // (log2(M) * F) tensor -> Binary tree
          char     * _treetrace,            // (log2(M) * F) tensor -> Binary tree
          offset_t   treesize,
    const offset_t * _size,                 // [*batch] list -> Size of `dist`
    const offset_t * _stride_gradcoord,     // [*batch] list -> Strides of `gradcoord`
    const offset_t * stride_gradvert,       // [N, D] list -> Strides of `gradvert`
    const offset_t * _stride_grad,          // [*batch] list -> Strides of `grad`
    const offset_t * _stride_coord,         // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces           // [M, D] list -> Strides of `faces`
)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    using Klass          = MeshDistCuda<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;

    auto treetrace = SizedStridedPointer<char, offset_t>(_treetrace + index, stride, treesize);

    StaticPoint<nbatch, offset_t> size;
    size.copy_(ConstRefPoint<nbatch, offset_t>(_size));
    StaticPoint<nbatch, offset_t> stride_grad;
    stride_grad.copy_(ConstRefPoint<nbatch, offset_t>(_stride_grad));
    StaticPoint<nbatch+1, offset_t> stride_coord;
    stride_coord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_coord));

    StaticPoint<nbatch+1, offset_t> stride_gradcoord;
    if (_stride_gradcoord)
        stride_gradcoord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_gradcoord));

    auto faces        = FaceList(_faces, stride_faces[0], stride_faces[1]);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto tree         = reinterpret_cast<const Node *>(_tree);

    offset_t numel = size.prod();
    for (offset_t i=index; index < numel; index += stride, i=index)
    {
        StaticPoint<ndim, scalar_t>             gradpoint;
        StaticPointList<ndim, ndim, scalar_t>   gradface;
        index_t                                 nearest_face;

        for (offset_t j=0; j<treetrace.size; ++j)
            treetrace[j] = static_cast<char>(0);

        offset_t offset_coord = index2offset<nbatch>(i, size.data, stride_coord.data);
        offset_t offset_grad  = index2offset<nbatch>(i, size.data, stride_grad.data);
        auto grad1 = grad[offset_grad];

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(
                coord + offset_coord, stride_coord[nbatch]
        ));

        dist[offset_dist] = Klass::unsigned_sqdist_grad(
            nearest_face,
            gradpoint,
            gradface,
            point,
            vertices, 
            faces, 
            tree,
            treetrace
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
    }
}

template <
    int nbatch,             // Number of batch dimensions in coord 
    int ndim,               // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
__global__ void dt_naive(
          scalar_t * dist,                  // (*batch) tensor -> Output placeholder for distance
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const offset_t * _size,                 // [*batch] list -> Size of `dist`
          offset_t   nb_faces,
    const offset_t * _stride_dist,          // [*batch] list -> Strides of `dist`
    const offset_t * _stride_coord,         // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces           // [M, D] list -> Strides of `faces`
)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    using Klass          = MeshDistCuda<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointListSized<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;


    StaticPoint<nbatch, offset_t> size;
    size.copy_(ConstRefPoint<nbatch, offset_t>(_size));
    StaticPoint<nbatch, offset_t> stride_dist;
    stride_dist.copy_(ConstRefPoint<nbatch, offset_t>(_stride_dist));
    StaticPoint<nbatch+1, offset_t> stride_coord;
    stride_coord.copy_(ConstRefPoint<nbatch+1, offset_t>(_stride_coord));

    auto faces        = FaceList(_faces, stride_faces[0], stride_faces[1], nb_faces);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);

    offset_t numel = size.prod();
    for (offset_t i=index; i < numel; i += stride)
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
    }
}