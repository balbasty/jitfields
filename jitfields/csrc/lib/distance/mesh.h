/*
 * Addapted from https://github.com/InteractiveComputerGraphics/TriangleMeshDistance
 */
#ifndef JF_DISTANCE_MESH_H
#define JF_DISTANCE_MESH_H
#include "../utils.h"
#include "mesh_utils.h"


namespace jf {
namespace distance_mesh {

// =============================================================================
//
//                       ENCODE ENTITIES IN A TRIANGLE
//
// =============================================================================

enum class NearestEntity: char {
    F, V0, V1, V2, E01, E02, E12
};

// =============================================================================
//
//                          2D/3D SPECIFIC FUNCTIONS
//
// =============================================================================

template <int D, typename scalar_t, typename offset_t>
class MeshDistUtil {};

// -----------------------------------------------------------------------------
//                              3D IMPLEMENTATION
// -----------------------------------------------------------------------------

template <typename scalar_t,  typename offset_t>
struct MeshDistUtil<3, scalar_t, offset_t> {

    template <typename Point, typename NearestPoint, typename Normals>
    __host__ __device__ static inline
    scalar_t sign(const Point & point, const NearestPoint & nearest_point, 
                  const Normals & pseudonormals, const NearestEntity & nearest_entity)
    {
        int i = 0;
        switch (nearest_entity)
        {
            case NearestEntity::F:
                i = 0;
                break;
            case NearestEntity::V0:
                i = 1;
                break;
            case NearestEntity::V1:
                i = 2;
                break;
            case NearestEntity::V2:
                i = 3;
                break;
            case NearestEntity::E01:
                i = 4;
                break;
            case NearestEntity::E12:
                i = 5;
                break;
            case NearestEntity::E02:
                i = 6;
                break;
            default:
                break;
        }
        auto pseudonormal = pseudonormals[i];


        auto u = point - nearest_point;
        scalar_t s = static_cast<scalar_t>(u.dot(pseudonormal) >= 0 ? 1 : -1);
        return s;
    }

    template <typename NearestPoint, typename Point, typename Vertices>
    __host__ __device__ static inline 
    scalar_t sqdist_unsigned(NearestEntity & nearest_entity, NearestPoint & nearest_point, 
                             const Point & point, const Vertices & vertices)
    {
        auto diff  = vertices[0] - point;
        auto edge0 = vertices[1] - vertices[0];
        auto edge1 = vertices[2] - vertices[0];
        auto a00   = edge0.dot(edge0);
        auto a01   = edge0.dot(edge1);
        auto a11   = edge1.dot(edge1);
        auto b0    = diff.dot(edge0);
        auto b1    = diff.dot(edge1);
        auto c     = diff.dot(diff);
        auto det   = abs(a00 * a11 - a01 * a01);
        auto s     = a01 * b1 - a11 * b0;
        auto t     = a01 * b0 - a00 * b1;
        auto d2    = static_cast<scalar_t>(-1);

        if (s + t <= det)
        {
            if (s < 0)
            {
                if (t < 0)
                {
                    if (b0 < 0)
                    {
                        t = static_cast<scalar_t>(0);
                        if (-b0 >= a00)
                        {
                            nearest_entity = NearestEntity::V1;
                            s = static_cast<scalar_t>(1);
                            d2 = a00 + 2 * b0 + c;
                        }
                        else
                        {
                            nearest_entity = NearestEntity::E01;
                            s = -b0 / a00;
                            d2 = b0 * s + c;
                        }
                    }
                    else
                    {
                        s = static_cast<scalar_t>(0);
                        if (b1 >= 0)
                        {
                            nearest_entity = NearestEntity::V0;
                            t = static_cast<scalar_t>(0);
                            d2 = c;
                        }
                        else if (-b1 >= a11)
                        {
                            nearest_entity = NearestEntity::V2;
                            t = static_cast<scalar_t>(1);
                            d2 = a11 + 2 * b1 + c;

                        }
                        else
                        {
                            nearest_entity = NearestEntity::E02;
                            t = -b1 / a11;
                            d2 = b1 * t + c;
                        }
                    }
                }
                else
                {
                    s = static_cast<scalar_t>(0);
                    if (b1 >= 0)
                    {
                        nearest_entity = NearestEntity::V0;
                        t = static_cast<scalar_t>(0);
                        d2 = c;
                    }
                    else if (-b1 >= a11)
                    {
                        nearest_entity = NearestEntity::V2;
                        t = static_cast<scalar_t>(1);
                        d2 = a11 + 2 * b1 + c;
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E02;
                        t = -b1 / a11;
                        d2 = b1 * t + c;
                    }
                }
            }
            else if (t < 0)
            {
                t = static_cast<scalar_t>(0);
                if (b0 >= 0)
                {
                    nearest_entity = NearestEntity::V0;
                    s = static_cast<scalar_t>(0);
                    d2 = c;
                }
                else if (-b0 >= a00)
                {
                    nearest_entity = NearestEntity::V1;
                    s = static_cast<scalar_t>(1);
                    d2 = a00 + 2 * b0 + c;
                }
                else
                {
                    nearest_entity = NearestEntity::E01;
                    s = -b0 / a00;
                    d2 = b0 * s + c;
                }
            }
            else
            {
                nearest_entity = NearestEntity::F;
                // minimum at interior point
                scalar_t invDet = 1 / det;
                s *= invDet;
                t *= invDet;
                d2 = s * (a00 * s + a01 * t + 2 * b0) +
                     t * (a01 * s + a11 * t + 2 * b1) + c;

            }
        }
        else
        {

            scalar_t tmp0, tmp1, numer, denom;

            if (s < 0)
            {
                tmp0 = a01 + b0;
                tmp1 = a11 + b1;
                if (tmp1 > tmp0)
                {
                    numer = tmp1 - tmp0;
                    denom = a00 - 2 * a01 + a11;
                    if (numer >= denom)
                    {
                        nearest_entity = NearestEntity::V1;
                        s = static_cast<scalar_t>(1);
                        t = static_cast<scalar_t>(0);
                        d2 = a00 + 2 * b0 + c;
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E12;
                        s = numer / denom;
                        t = 1 - s;
                        d2 = s * (a00 * s + a01 * t + 2 * b0) +
                             t * (a01 * s + a11 * t + 2 * b1) + c;
                    }
                }
                else
                {
                    s = static_cast<scalar_t>(0);
                    if (tmp1 <= 0)
                    {
                        nearest_entity = NearestEntity::V2;
                        t = static_cast<scalar_t>(1);
                        d2 = a11 + 2 * b1 + c;
                    }
                    else if (b1 >= 0)
                    {
                        nearest_entity = NearestEntity::V0;
                        t = static_cast<scalar_t>(0);
                        d2 = c;
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E02;
                        t = -b1 / a11;
                        d2 = b1 * t + c;
                    }
                }
            }
            else if (t < 0)
            {
                tmp0 = a01 + b1;
                tmp1 = a00 + b0;
                if (tmp1 > tmp0)
                {
                    numer = tmp1 - tmp0;
                    denom = a00 - 2 * a01 + a11;
                    if (numer >= denom)
                    {
                        nearest_entity = NearestEntity::V2;
                        t = static_cast<scalar_t>(1);
                        s = static_cast<scalar_t>(0);
                        d2 = a11 + 2 * b1 + c;
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E12;
                        t = numer / denom;
                        s = 1 - t;
                        d2 = s * (a00 * s + a01 * t + 2 * b0) +
                             t * (a01 * s + a11 * t + 2 * b1) + c;
                    }
                }
                else
                {
                    t = static_cast<scalar_t>(0);
                    if (tmp1 <= 0)
                    {
                        nearest_entity = NearestEntity::V1;
                        s = static_cast<scalar_t>(1);
                        d2 = a00 + 2 * b0 + c;
                    }
                    else if (b0 >= 0)
                    {
                        nearest_entity = NearestEntity::V0;
                        s = static_cast<scalar_t>(0);
                        d2 = c;
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E01;
                        s = -b0 / a00;
                        d2 = b0 * s + c;
                    }
                }
            }
            else
            {
                numer = a11 + b1 - a01 - b0;
                if (numer <= 0)
                {
                    nearest_entity = NearestEntity::V2;
                    s = static_cast<scalar_t>(0);
                    t = static_cast<scalar_t>(1);
                    d2 = a11 + 2 * b1 + c;
                }
                else
                {
                    denom = a00 - 2 * a01 + a11;
                    if (numer >= denom)
                    {
                        nearest_entity = NearestEntity::V1;
                        s = static_cast<scalar_t>(1);
                        t = static_cast<scalar_t>(0);
                        d2 = a00 + 2 * b0 + c;
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E12;
                        s = numer / denom;
                        t = 1 - s;
                        d2 = s * (a00 * s + a01 * t + 2 * b0) +
                             t * (a01 * s + a11 * t + 2 * b1) + c;
                    }
                }
            }
        }

        for (int d=0; d<3; ++d)
            nearest_point[d] = vertices[0][d] + s * edge0[d] + t * edge1[d];

        // Account for numerical round-off error.
        if (d2 < 0) {
            printf("neg dist: %f true dist: %f entity = %d\n", d2, (nearest_point - point).sqnorm(), nearest_entity);
            d2 = static_cast<scalar_t>(0);
        }
        return d2;
    }

    // Returns pseudonormals ordered as: F, V0, V1, V2
    template <typename Normals, typename Triangle>
    __host__ __device__ static inline
    void compute_pseudonormals(Normals & pseudonormals, const Triangle & triangle)
    {
        // face
        pseudonormals[0].crossto_(triangle[1] - triangle[0], triangle[2] - triangle[0]);
        pseudonormals[0].normalize_();

        // vertex 0
        {
            auto b_minus_a = triangle[1] - triangle[0]; b_minus_a.normalize_();
            auto c_minus_a = triangle[2] - triangle[0]; c_minus_a.normalize_();
            auto alpha = std::acos(b_minus_a.dot(c_minus_a));
            pseudonormals[1].copy_(pseudonormals[0], alpha);
        }

        // vertex 1
        {
            auto a_minus_b = triangle[0] - triangle[1]; a_minus_b.normalize_();
            auto c_minus_b = triangle[2] - triangle[1]; c_minus_b.normalize_();
            auto alpha = std::acos(a_minus_b.dot(c_minus_b));
            pseudonormals[2].copy_(pseudonormals[0], alpha);
        }

        // vertex 2
        {
            auto b_minus_c = triangle[1] - triangle[2]; b_minus_c.normalize_();
            auto a_minus_c = triangle[0] - triangle[2]; a_minus_c.normalize_();
            auto alpha = std::acos(b_minus_c.dot(a_minus_c));
            pseudonormals[3].copy_(pseudonormals[0], alpha);
        }
    }

};

// =============================================================================
//
//                                  GENERIC API
//
// =============================================================================

template <int D, typename scalar_t, typename index_t, typename offset_t>
struct MeshDist {
    static constexpr offset_t ndim  = static_cast<offset_t>(D);
    using Utils = MeshDistUtil<D, scalar_t, offset_t>;
    using StaticPointScalar = StaticPoint<D, scalar_t>;

    struct BoundingSphere {
        virtual ~BoundingSphere() {}

        StaticPointScalar center;
        scalar_t radius;
    };

    struct Node {
        virtual ~Node() {}

        BoundingSphere bv_left;
        BoundingSphere bv_right;
        index_t left   = -1;
        index_t right  = -1;
        index_t parent = -1;
    };

    template <typename Face>
    __host__ __device__ static inline
    BoundingSphere bounding_sphere(const Face & face)
    {
        BoundingSphere sphere;

        sphere.center.addto_(face[0], face[1]);
        if (D == 3)
            sphere.center += face[2];
        sphere.center /= static_cast<scalar_t>(D);

        sphere.radius = (face[0] - sphere.center).norm();
        for (offset_t d=1; d<D; ++d)
        {
            sphere.radius = max(sphere.radius, (face[d] - sphere.center).norm());
        }

        return sphere;
    }

#ifndef __CUDACC__

    // This logic is overly complex, but it's the only way I managed to 
    // get std::sort to work on a strided array without copying the 
    // entire array to contiguous memory.

    struct Face {

        using Ref = StridedPoint<D, index_t, offset_t>;
        using Copy = StaticPoint<D, index_t>;

        Face(index_t * ptr, offset_t stride): face(ptr, stride), copy(face) {}
        Face(const Face & other): face(other.face.data, other.face.stride), copy(face) {}

        Face & operator= (const Face & other)
        {
            face.copy_(other.copy);
            copy.copy_(other.copy);
            return *this;
        }

        Face & operator+= (offset_t offset) { face.data += offset; return *this; }
        Face & operator-= (offset_t offset) { face.data -= offset; return *this; }

        Face & change_(const Face & other)
        {
            face.data = other.face.data;
            face.stride = other.face.stride;
            copy.copy_(other.copy);
            return *this;
        }

        Face & load_()
        {
            copy.copy_(face);
            return *this;
        }

        Face load()
        {
            auto clone = Face(*this);
            clone.load_();
            return *this;
        }

        Ref face;
        Copy copy;
    };

    struct FaceIterator {
        
        using this_type = FaceIterator;
        using difference_type = offset_t;
        using value_type = Face;
        using reference = value_type&;
        using pointer = value_type*;
        using iterator_category = std::random_access_iterator_tag;

        FaceIterator(index_t * elem, offset_t stride, offset_t stridein):
            ptr(elem, stridein), stride(stride) {}

        FaceIterator(const this_type & other):
            ptr(other.ptr.face.data, other.ptr.face.stride), stride(other.stride) {}
        
        this_type & operator= (const this_type & other) 
            { ptr.change_(other.ptr); stride = other.stride; 
              return *this; }

        value_type & operator* () { return ptr.load_(); }
        value_type operator* () const { return ptr.load(); } 
        value_type & operator[] (difference_type n) { return *(*this + n); }
        value_type operator[] (difference_type n) const { return *(*this + n); } 

        this_type & operator ++ () { ptr += stride; return *this; }
        this_type & operator -- () { ptr -= stride; return *this; }
        this_type operator ++ (int) { this_type prev = *this; ptr += stride; return prev; }
        this_type operator -- (int) { this_type prev = *this; ptr -= stride; return prev; }

        friend this_type & operator += (this_type & a, difference_type n) { a.ptr += n * a.stride; return a; }
        friend this_type & operator -= (this_type & a, difference_type n) { a.ptr -= n * a.stride; return a; }
        friend this_type & operator += (this_type & a, const this_type & b) { a.ptr += b.ptr.face.data; return a; }
        friend this_type & operator -= (this_type & a, const this_type & b) { a.ptr -= b.ptr.face.data; return a; }
        friend this_type operator + (const this_type & a, difference_type n) { return this_type(a.ptr.face.data + n * a.stride, a.stride, a.ptr.face.stride); }  
        friend this_type operator + (difference_type n, const this_type & a) { return this_type(a.ptr.face.data + n * a.stride, a.stride, a.ptr.face.stride); }
        friend this_type operator - (const this_type & a, difference_type n) { return this_type(a.ptr.face.data - n * a.stride, a.stride, a.ptr.face.stride); } 
        friend difference_type operator + (const this_type & a, const this_type & b) { return static_cast<offset_t>(a.ptr.face.data + b.ptr.face.data) / a.stride; }
        friend difference_type operator - (const this_type & a, const this_type & b) { return static_cast<offset_t>(a.ptr.face.data - b.ptr.face.data) / a.stride; }
        
        friend bool operator== (const this_type & a, const this_type & b) { return a.ptr.face.data == b.ptr.face.data; } 
        friend bool operator!= (const this_type & a, const this_type & b) { return a.ptr.face.data != b.ptr.face.data; } 
        friend bool operator<  (const this_type & a, const this_type & b) { return a.stride > 0 ? a.ptr.face.data <  b.ptr.face.data : a.ptr.face.data >  b.ptr.face.data; } 
        friend bool operator>  (const this_type & a, const this_type & b) { return a.stride > 0 ? a.ptr.face.data >  b.ptr.face.data : a.ptr.face.data <  b.ptr.face.data; } 
        friend bool operator<= (const this_type & a, const this_type & b) { return a.stride > 0 ? a.ptr.face.data <= b.ptr.face.data : a.ptr.face.data >= b.ptr.face.data; } 
        friend bool operator>= (const this_type & a, const this_type & b) { return a.stride > 0 ? a.ptr.face.data >= b.ptr.face.data : a.ptr.face.data <= b.ptr.face.data; } 

        value_type ptr;
        offset_t stride = static_cast<offset_t>(1); // stride between two faces
    };

    template <typename Faces, typename Vertices>
    static inline
    BoundingSphere build_tree(
        Node           * nodes, 
        index_t        & node_id, 
        index_t          parent_id,
        index_t          begin, 
        index_t          end, 
        Faces          & faces, 
        const Vertices & vertices)
    {
        offset_t nb_faces = end - begin;
        if (nb_faces == 0) 
        {
            // not normal...
            return BoundingSphere();
        }
        else if (nb_faces == 1)
        {
            // leaf
            nodes[node_id].parent = parent_id;
            nodes[node_id].left = -1;
            nodes[node_id].right = begin;

            auto face = faces[begin];
            auto facevertices = StaticPointList<D, D, scalar_t>();
            for (offset_t d=0; d<D; ++d)
                facevertices[d].copy_(vertices[face[d]]);

            return bounding_sphere(facevertices);
        }
        else
        {
		    // Compute AxisAligned Bounding Box center and largest dimension of all current triangles
            StaticPointScalar top, bottom, center;
            top.copy_(static_cast<scalar_t>(-1./0.));
            bottom.copy_(static_cast<scalar_t>(1./0.));
            center.copy_(static_cast<scalar_t>(0.));
            for (index_t i = begin; i < end; ++i)
            {
                auto face = faces[i];
                StaticPointScalar v;
                for (offset_t d=0; d<D; ++d)
                {
                    v.copy_(vertices[face[d]]);
                    center.add_(v);
                    top.max_(v);
                    bottom.min_(v);
                }
            }
            center /= static_cast<scalar_t>(D*nb_faces);
            top -= bottom;
            offset_t split_dim;
            if (D == 3)
                split_dim = top[0] > top[1] ? (top[0] > top[2] ? 0 : 2) : (top[1] > top[2] ? 1 : 2);
            else // D == 2
                split_dim = top[0] > top[1] ? 0 : 1;

            scalar_t radius_sq = 0;
            for (index_t i=begin; i < end; ++i)
            {
                auto face = faces[i];
                for (offset_t d=0; d<D; ++d)
                {
                    auto v = vertices[face[d]] - center;
                    radius_sq = max(radius_sq, v.sqnorm());
                }
            }
            BoundingSphere sphere;
            sphere.center = center;
            sphere.radius = sqrt(radius_sq);

            std::sort(
                FaceIterator(faces.data, faces.stride_elem, faces.stride_channel) + begin,
                FaceIterator(faces.data, faces.stride_elem, faces.stride_channel) + end,
                [&](const Face & a, const Face & b) {
                    return vertices[a.face[0]][split_dim] < vertices[b.face[0]][split_dim];
                });

            index_t mid = (begin + end) / 2;
            index_t current_id = node_id;
            nodes[current_id].parent = parent_id;

            node_id += 1;
            nodes[current_id].left = node_id;
            nodes[current_id].bv_left = build_tree(
                nodes, node_id, current_id, begin, mid, faces, vertices);

            node_id += 1;
            nodes[current_id].right = node_id;
            nodes[current_id].bv_right = build_tree(
                nodes, node_id, current_id, mid, end, faces, vertices);

            return sphere;
        }
    }

    template <typename NormFaces, typename NormVertices, typename NormEdges, 
              typename Faces, typename Vertices>
    static inline 
    void build_normals(
        NormFaces       & normfaces, 
        NormVertices    & normvertices, 
        NormEdges       & normedges, 
        const Faces     & faces,
        const Vertices  & vertices)
    {
        std::unordered_map<offset_t, StaticPointScalar> normedges_dict;

        auto get_edge_id = [&](offset_t i, offset_t j) {
            return min(i, j) * vertices.size() + max(i, j);
        };

        for (offset_t n=0; n<faces.size(); ++n)
        {
            auto face = faces[n];

            // compute normals
            auto normals      = StaticPointList<D+1, D, scalar_t>();
            auto facevertices = StaticPointList<D, D, scalar_t>();
            auto vertex_id    = StaticPoint<D, offset_t>();
            for (offset_t d=0; d<D; ++d)
            {
                vertex_id[d]    = static_cast<offset_t>(face[d]);
                facevertices[d].copy_(vertices[vertex_id[d]]);
            }
            Utils::compute_pseudonormals(normals, facevertices);

            // accumulate normals
            normfaces[n].copy_(normals[0]);
            for (offset_t d=0; d<D; ++d)
            {
                normvertices[vertex_id[d]].add_(normals[d+1]);
                if (D == 3)
                {
                    index_t edge_id = (d == 0 ? get_edge_id(vertex_id[0], vertex_id[1]): 
                                       d == 1 ? get_edge_id(vertex_id[1], vertex_id[2]): 
                                                get_edge_id(vertex_id[0], vertex_id[2]));
                    if (normedges_dict.find(edge_id) == normedges_dict.end())
                    {
                        normedges_dict[edge_id] = StaticPointScalar(normals[0]);
                    }
                    else
                    {
                        normedges_dict[edge_id].add_(normals[0]);
                    }
                }
            }
        }

        // normalize
        for (offset_t n=0; n<vertices.size(); ++n)
            normvertices[n].normalize_();

        // build final edge map
        if (D  == 3) 
        {
            for (auto edge = normedges_dict.begin(); edge != normedges_dict.end(); ++edge)
                edge->second.normalize_();
            for (offset_t n=0; n<faces.size(); ++n)
            {
                auto face = faces[n];
                auto vertex_id = StaticPoint<D, offset_t>();
                for (offset_t d=0; d<D; ++d)
                {
                    vertex_id[d] = static_cast<offset_t>(face[d]);
                }

                for (offset_t d=0; d<D; ++d)
                {
                    index_t edge_id = (d == 0 ? get_edge_id(vertex_id[0], vertex_id[1]): 
                                       d == 1 ? get_edge_id(vertex_id[1], vertex_id[2]): 
                                                get_edge_id(vertex_id[0], vertex_id[2]));
                    normedges[n][d].copy_(normedges_dict[edge_id]);
                }
            }
        }
    }

#endif

    template <typename NearestPoint, typename Point, typename Vertices, typename Faces> 
    __host__ __device__ static inline
    void query_dist_recurse(
            index_t        & nearest_face,
            scalar_t       & nearest_dist,
            NearestEntity  & nearest_entity,
            NearestPoint   & nearest_point,
            index_t          node_id,
            const Point    & point, 
            const Vertices & vertices, 
            const Faces    & faces, 
            const Node     * nodes)
    {
        auto node = nodes[node_id];
        if (node.left == -1)
        {
            // leaf

            const offset_t face_id = static_cast<offset_t>(node.right);
            auto face = faces[face_id];
            auto facevertices = StaticPointList<D, D, scalar_t>();
            for (offset_t d=0; d<D; ++d)
                facevertices[d].copy_(vertices[face[d]]);
            
            NearestEntity       maybe_entity;
            StaticPointScalar   maybe_point;
            scalar_t maybe_dist = Utils::sqdist_unsigned(maybe_entity, maybe_point, point, facevertices);

            if (maybe_dist < nearest_dist * nearest_dist)
            {
                nearest_face   = face_id;
                nearest_dist   = sqrt(maybe_dist);
                nearest_entity = maybe_entity;
                nearest_point  = maybe_point;
            }
        }
        else
        {
            // find which child bounding volume is closer
            const scalar_t d_left  = (point - node.bv_left.center).norm()  - node.bv_left.radius;
            const scalar_t d_right = (point - node.bv_right.center).norm() - node.bv_right.radius;
            
            auto do_query = [&](index_t next_node_id) {
                return query_dist_recurse(
                    nearest_face,
                    nearest_dist,
                    nearest_entity,
                    nearest_point,
                    next_node_id,
                    point, 
                    vertices, 
                    faces, 
                    nodes);
            };

            if (d_left < d_right) 
            {
                if (d_left < nearest_dist)  do_query(node.left);
                if (d_right < nearest_dist) do_query(node.right);
            }
            else 
            {
                if (d_right < nearest_dist) do_query(node.right);
                if (d_left < nearest_dist)  do_query(node.left);
            }
        }
    }

    template <typename NearestPoint, typename Point, typename Vertices, typename Faces, typename Trace> 
    __host__ __device__ static inline
    void query_dist_loop(
            index_t        & nearest_face,
            scalar_t       & nearest_dist,
            NearestEntity  & nearest_entity,
            NearestPoint   & nearest_point,
            const Point    & point, 
            const Vertices & vertices, 
            const Faces    & faces, 
            const Node     * nodes,
            Trace          & trace)
    {
        const Node * node = nullptr;
        index_t node_id = 0;
        index_t level = 0;

        // we use the first four bits of trace (at each level) as follow:
        // higher bit -> lower bit
        // [current_side_is_right, current_side_is_left, right_was_visited, left_was_visited]
        //
        // Therefore
        // left_was_visited  == trace & 1
        // right_was_visited == trace & 2
        // current_side_is_left  == trace & 4 == (trace >> 2) & 1
        // current_side_is_right == trace & 5 == (trace >> 2) & 2

        auto fast_dist = [&](const BoundingSphere & sphere)
        {
            scalar_t dist = 0;
            for (int d=0; d<D; ++d)
            {
                scalar_t tmp = point[d] - sphere.center[d];
                dist += tmp*tmp;
            }
            dist = sqrt(dist);
            dist -= sphere.radius;
            return dist;
        };

        while (1)
        {
            if (node_id < 0)
                break;
            
            node = nodes + node_id;

            if (node->left == -1)
            {
                // leaf

                const offset_t face_id = static_cast<offset_t>(node->right);
                auto face = faces[face_id];
                auto facevertices = StaticPointList<D, D, scalar_t>();
                for (offset_t d=0; d<D; ++d)
                    facevertices[d].copy_(vertices[face[d]]);
                
                NearestEntity       maybe_entity;
                StaticPointScalar   maybe_point;
                scalar_t maybe_dist = Utils::sqdist_unsigned(maybe_entity, maybe_point, point, facevertices);

                if (maybe_dist < nearest_dist * nearest_dist)
                {
                    nearest_face   = face_id;
                    nearest_dist   = sqrt(maybe_dist);
                    nearest_entity = maybe_entity;
                    nearest_point.copy_(maybe_point);
                }
                level -= 1;
                trace[level] |= trace[level] >> 2; // set current side as "visited"
                node_id = node->parent;
            }
            else if ((trace[level] & 1) && (trace[level] & 2))
            {
                // left and right already visited
                if (level == 0)
                    break;
                for (index_t l=level; l<trace.size; ++l)
                    trace[l] = 0;
                node_id = node->parent;
                level -= 1;
                trace[level] |= trace[level] >> 2; // set current side as "visited"
            }
            else if(trace[level] & 2)
            {
                // already visited right, now visit left
                const scalar_t d_left  = fast_dist(node->bv_left);

                if (d_left < nearest_dist)
                {
                    trace[level] &= 3;      // erase side bits
                    trace[level] |= 1 << 2; // set current side as "left"
                    node_id = node->left;
                    level += 1;
                    continue;
                } 
                else 
                {
                    trace[level] |= 1;     // set left as "visited"
                }
            }
            else if(trace[level] & 1)
            {
                // already visited left, now visit right
                const scalar_t d_right = fast_dist(node->bv_right);

                if (d_right < nearest_dist)
                {
                    trace[level] &= 3;      // erase side bits
                    trace[level] |= 2 << 2; // set current side as "right"
                    node_id = node->right;
                    level += 1;
                    continue;
                } 
                else 
                {
                    trace[level] |= 2;     // set right as "visited"
                }
            }
            else
            {
                // none visited - decide whether to start with left or right
                scalar_t d_left  = fast_dist(node->bv_left);
                scalar_t d_right = fast_dist(node->bv_right);

                if (d_left < d_right) 
                {
                    if (d_left < nearest_dist)
                    {
                        trace[level] &= 3;      // erase side bits
                        trace[level] |= 1 << 2; // set current side as "left"
                        node_id = node->left;
                        level += 1;
                        continue;
                    } 
                    else 
                    {
                        trace[level] |= 1;     // set left as "visited"
                        if (d_right < nearest_dist)
                        {
                            trace[level] &= 3;      // erase side bits
                            trace[level] |= 2 << 2; // set current side as "right"
                            node_id = node->right;
                            level += 1;
                            continue;
                        }
                        else
                        {
                            trace[level] |= 2;     // set right as "visited"
                        }
                    }
                }
                else 
                {
                    if (d_right < nearest_dist)
                    {
                        trace[level] &= 3;      // erase side bits
                        trace[level] |= 2 << 2; // set current side as "right"
                        node_id = node->right;
                        level += 1;
                        continue;
                    } 
                    else 
                    {
                        trace[level] |= 2;     // set right as "visited"
                        if (d_left < nearest_dist)
                        {
                            trace[level] &= 3;      // erase side bits
                            trace[level] |= 1 << 2; // set current side as "left"
                            node_id = node->left;
                            level += 1;
                            continue;
                        }
                        else
                        {
                            trace[level] |= 1;     // set left as "visited"
                        }
                    }
                }
            }
        }
    }

// #define DIST_USE_LOOP 1
#ifdef __CUDACC__
#define DIST_USE_LOOP
#endif


    template <typename NearestPoint, typename Point, typename Vertices, typename Faces
#ifdef DIST_USE_LOOP
    ,typename Trace
#endif
    > 
    __host__ __device__ static inline
    scalar_t _unsigned_dist(
            index_t        & nearest_face,
            NearestEntity  & nearest_entity,
            NearestPoint   & nearest_point,
            const Point    & point, 
            const Vertices & vertices, 
            const Faces    & faces, 
            const Node     * tree
#ifdef DIST_USE_LOOP
            , Trace        & treetrace
#endif
    )
    {
        scalar_t nearest_dist = static_cast<scalar_t>(1./0.);
#ifdef DIST_USE_LOOP
        query_dist_loop(
#else
        query_dist_recurse(
#endif
            nearest_face,
            nearest_dist,
            nearest_entity,
            nearest_point,
#ifndef DIST_USE_LOOP
            static_cast<index_t>(0),
#endif
            point, 
            vertices, 
            faces, 
            tree
#ifdef DIST_USE_LOOP
            ,treetrace
#endif
        );
        return nearest_dist;
    }

    template <typename Point, typename Vertices, typename Faces
#ifdef DIST_USE_LOOP
    ,typename Trace
#endif
    > 
    __host__ __device__ static inline
    scalar_t unsigned_dist(
            const Point    & point, 
            const Vertices & vertices, 
            const Faces    & faces, 
            const Node     * tree
#ifdef DIST_USE_LOOP
            , Trace        & treetrace
#endif
    )
    {
        index_t             nearest_face;
        StaticPointScalar   nearest_point;
        NearestEntity       nearest_entity;

        return _unsigned_dist(
            nearest_face,
            nearest_entity,
            nearest_point,
            point,
            vertices, 
            faces, 
            tree
#ifdef DIST_USE_LOOP
            ,treetrace
#endif
        );
    }

    template <typename Point, typename Vertices, typename Faces, 
              typename NormFaces, typename NormEdges, typename NormVertices
#ifdef DIST_USE_LOOP
    ,typename Trace
#endif
    >  
    __host__ __device__ static inline
    scalar_t signed_dist(
            const Point        & point, 
            const Vertices     & vertices, 
            const Faces        & faces, 
            const Node         * tree, 
#ifdef DIST_USE_LOOP
            Trace              & treetrace,
#endif
            const NormFaces    & normfaces, 
            const NormEdges    & normedges, 
            const NormVertices & normvertices
    )
    {
        index_t             nearest_face;
        StaticPointScalar   nearest_point;
        NearestEntity       nearest_entity;

        // compute unsigned distance and return index of nearest triangle
        scalar_t dist = _unsigned_dist(
            nearest_face,
            nearest_entity,
            nearest_point,
            point,
            vertices, 
            faces, 
            tree
#ifdef DIST_USE_LOOP
            ,treetrace
#endif
        );

        // load normals into a compact array
        auto normals     = StaticPointList<D+1+(D == 3 ? D : 0), D, scalar_t>();
        auto face        = faces[nearest_face];
        auto normedge    = normedges[nearest_face];
        normals[0].copy_(normfaces[nearest_face]);
        for (offset_t d=0; d<D; ++d)
        {
            normals[1+d].copy_(normvertices[face[d]]);
            if (D == 3) { normals[1+D+d].copy_(normedge[d]); }
        }

        // compute sign from dot product <ray, normal>
        scalar_t sign = Utils::sign(
            point, 
            nearest_point, 
            normals, 
            nearest_entity
        );

        return dist * sign;
    }

    template <typename NearestPoint, typename Point, typename Vertices, typename Faces> 
    __host__ __device__ static inline
    scalar_t _unsigned_dist_naive(
            index_t        & nearest_face,
            NearestEntity  & nearest_entity,
            NearestPoint   & nearest_point,
            const Point    & point, 
            const Vertices & vertices, 
            const Faces    & faces
    )
    {
        scalar_t nearest_dist = static_cast<scalar_t>(1./0.);
        for (offset_t n=0; n<faces.size(); ++n)
        {
            NearestEntity       maybe_entity;
            StaticPointScalar   maybe_point;
            scalar_t            maybe_dist;
            {
                auto face = faces[n];
                auto facevertices = StaticPointList<D, D, scalar_t>();
                for (offset_t d=0; d<D; ++d)
                    facevertices[d].copy_(vertices[face[d]]);
                
                maybe_dist = Utils::sqdist_unsigned(maybe_entity, maybe_point, point, facevertices);

                if (maybe_dist == 0)
                {
                    printf("zero dist: point = [%f %f %f] nearest = [%f %f %f] vertices = [%f %f %f; %f %f %f; %f %f %f] entity = %d\n",
                           point[0], point[1], point[2], 
                           maybe_point[0], maybe_point[1], maybe_point[2],
                           facevertices[0][0], facevertices[0][1], facevertices[0][2], 
                           facevertices[1][0], facevertices[1][1], facevertices[1][2], 
                           facevertices[2][0], facevertices[2][1], facevertices[2][2],
                           maybe_entity);
                }
            }

            if (maybe_dist < nearest_dist * nearest_dist)
            {
                nearest_face   = n;
                nearest_dist   = jf::sqrt(maybe_dist);
                nearest_entity = maybe_entity;
                nearest_point  = maybe_point;
            }
        }

        return nearest_dist;
    }

    template <typename Point, typename Vertices, typename Faces> 
    __host__ __device__ static inline
    scalar_t unsigned_dist_naive(
            const Point    & point, 
            const Vertices & vertices, 
            const Faces    & faces
    )
    {
        index_t             nearest_face;
        StaticPointScalar   nearest_point;
        NearestEntity       nearest_entity;

        return _unsigned_dist_naive(
            nearest_face,
            nearest_entity,
            nearest_point,
            point,
            vertices, 
            faces
        );
    }

    template <typename Point, typename Vertices, typename Faces, 
              typename NormFaces, typename NormEdges, typename NormVertices> 
    __host__ __device__ static inline
    scalar_t signed_dist_naive(
            const Point        & point, 
            const Vertices     & vertices, 
            const Faces        & faces, 
            const NormFaces    & normfaces, 
            const NormEdges    & normedges, 
            const NormVertices & normvertices
    )
    {
        index_t             nearest_face;
        StaticPointScalar   nearest_point;
        NearestEntity       nearest_entity;

        // compute unsigned distance and return index of nearest triangle
        scalar_t dist = _unsigned_dist_naive(
            nearest_face,
            nearest_entity,
            nearest_point,
            point,
            vertices, 
            faces
        );

        // load normals into a compact array
        auto normals     = StaticPointList<D+1+(D == 3 ? D : 0), D, scalar_t>();
        auto face        = faces[nearest_face];
        auto normedge    = normedges[nearest_face];
        normals[0].copy_(normfaces[nearest_face]);
        for (offset_t d=0; d<D; ++d)
        {
            normals[1+d].copy_(normvertices[face[d]]);
            if (D == 3) normals[1+D+d].copy_(normedge[d]);
        }

        // compute sign from dot product <ray, normal>
        scalar_t sign = Utils::sign(
            point, 
            nearest_point, 
            normals, 
            nearest_entity
        );

        return dist * sign;
    }
};

} // namespace distance_mesh
} // namespace jf

#endif // JF_DISTANCE_MESH_H