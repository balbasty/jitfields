#ifndef JF_DISTANCE_MESH_CPP_H
#define JF_DISTANCE_MESH_CPP_H
#include "../utils.h"
#include "mesh_utils.h"
#include "mesh.h"


namespace jf {
namespace distance_mesh {


template <int D, typename scalar_t, typename offset_t>
struct MeshDistUtilCpp: public MeshDistUtil<D, scalar_t, offset_t> {};

// -----------------------------------------------------------------------------
//                              2D IMPLEMENTATION
// -----------------------------------------------------------------------------

template <typename scalar_t,  typename offset_t>
struct MeshDistUtilCpp<2, scalar_t, offset_t>: 
    public MeshDistUtil<2, scalar_t, offset_t>
{
    static constexpr int D = 2;

    // Compute the normal to a face
    //
    // Output Arguments
    // ----------------
    // normal (Point2d) : Normal to a segment
    //
    // Input Arguments
    // ---------------
    // vertices (2-List of Point2d) : Endpoints of the segment
    template <typename Normal, typename Vertices>
    __host__ __device__ static inline
    void compute_normal(Normal & normal, const Vertices & vertices)
    {
        auto edge = vertices[1] - vertices[0];
        normal[0] = -edge[1];
        normal[1] =  edge[0];
        normal.normalize_();
    }

    // Compute the normal to a face
    //
    // Output Arguments
    // ----------------
    // normfaces (List of Point2d) : Normals to each face
    // normvertices (List of Point2d) : Normals to each vertex
    // [unused] normedges (List of Point2d)
    //
    // Input Arguments
    // ---------------
    // faces (List of Point2d) : Vertex indices of each segment
    // vertices (List of Point2d) : Vertex coordinates
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
        auto get_edge_id = [&](offset_t i, offset_t j) {
            return min(i, j) * vertices.size() + max(i, j);
        };

        for (offset_t n=0; n<faces.size(); ++n)
        {
            auto face = faces[n];

            // compute normals
            auto normal       = StaticPoint<D, scalar_t>();
            auto facevertices = StaticPointList<D, D, scalar_t>();
            auto vertex_id    = StaticPoint<D, offset_t>();
            for (offset_t d=0; d<D; ++d)
            {
                vertex_id[d] = static_cast<offset_t>(face[d]);
                facevertices[d].copy_(vertices[vertex_id[d]]);
            }
            compute_normal(normal, facevertices);

            // accumulate normals
            normfaces[n].copy_(normal);
            for (offset_t d=0; d<D; ++d)
                normvertices[vertex_id[d]].add_(normal);
        }

        // normalize
        for (offset_t n=0; n<vertices.size(); ++n)
            normvertices[n].normalize_();
    }
};

// -----------------------------------------------------------------------------
//                              3D IMPLEMENTATION
// -----------------------------------------------------------------------------

template <typename scalar_t,  typename offset_t>
struct MeshDistUtilCpp<3, scalar_t, offset_t>: 
    public MeshDistUtil<2, scalar_t, offset_t>
{
    static constexpr int D = 3;

    // Returns pseudonormals ordered as: F, V0, V1, V2
    //
    // Output Arguments
    // ----------------
    // pseudonormals (4-List of Point3d) : Normals to the face and each vertex
    //
    // Input Arguments
    // ---------------
    // triangle (3-List of Point3d) : Vertices of a triangle
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

    // Compute the normal to a face
    //
    // Output Arguments
    // ----------------
    // normfaces (List of Point2d) : Normals to each face
    // normvertices (List of Point2d) : Normals to each vertex
    // normedges (List of Point2d) : Normals to each edge
    //
    // Input Arguments
    // ---------------
    // faces (List of Point3d) : Vertex indices of each segment
    // vertices (List of Point3d) : Vertex coordinates
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
        std::unordered_map<offset_t, StaticPoint<D, scalar_t> > normedges_dict;

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
            compute_pseudonormals(normals, facevertices);

            // accumulate normals
            normfaces[n].copy_(normals[0]);
            for (offset_t d=0; d<D; ++d)
            {
                normvertices[vertex_id[d]].add_(normals[d+1]);
                auto edge_id = (d == 0 ? get_edge_id(vertex_id[0], vertex_id[1]): 
                                d == 1 ? get_edge_id(vertex_id[1], vertex_id[2]): 
                                         get_edge_id(vertex_id[0], vertex_id[2]));
                if (normedges_dict.find(edge_id) == normedges_dict.end())
                {
                    normedges_dict[edge_id] = StaticPoint<D, scalar_t>(normals[0]);
                }
                else
                {
                    normedges_dict[edge_id].add_(normals[0]);
                }
            }
        }

        // normalize
        for (offset_t n=0; n<vertices.size(); ++n)
            normvertices[n].normalize_();

        // build final edge map
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
                auto edge_id = (d == 0 ? get_edge_id(vertex_id[0], vertex_id[1]): 
                                d == 1 ? get_edge_id(vertex_id[1], vertex_id[2]): 
                                         get_edge_id(vertex_id[0], vertex_id[2]));
                normedges[n][d].copy_(normedges_dict[edge_id]);
            }
        }
    }
};

// =============================================================================
//
//                                  GENERIC API
//
// =============================================================================


template <int D, typename scalar_t, typename index_t, typename offset_t>
struct MeshDistCpp: public MeshDist<D, scalar_t, index_t, offset_t> {
    using Base              = MeshDist<D, scalar_t, index_t, offset_t>;
    using Utils             = MeshDistUtilCpp<D, scalar_t, offset_t>;
    using StaticPointScalar = StaticPoint<D, scalar_t>;
    using BoundingSphere    = typename Base::BoundingSphere;
    using Node              = typename Base::Node;

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
            { ptr.change_(other.ptr); stride = other.stride; return *this; }

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

    // Build a search tree over faces
    //
    // Ouput Arguments
    // ---------------
    // nodes (*Node) : search tree
    //
    // Input/Output Arguments
    // ----------------------
    // node_id (integer) : current index in the tree (initially zero)
    // faces : (List of Point) : All faces, will get progressively sorted
    //
    // Input Arguments
    // ---------------
    // parent_id (integer) : Index of the parent node in the tree
    // begin (integer) : Index of the first face in the current range
    // end (integer) : Index of the first face *not* in the current range
    // vertices (List of Point) : All vertices coordinates
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
            // should not happen...
            return BoundingSphere();
        }
        else if (nb_faces == 1)
        {
            // leaf
            nodes[node_id].parent = parent_id;
            nodes[node_id].left = -1;
            nodes[node_id].right = begin;

            auto facevertices = Base::get_facevertices(begin, vertices, faces);
            return Base::bounding_sphere(facevertices);
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

    // Find the face nearest to a target point
    //
    // On the cpu, we can recursively traverse the tree.
    // The point of the tree search is that we can cut long branches that 
    // we know are already too far.
    //
    // Output Arguments
    // ----------------
    // nearest_face     (integer) : Index of the nearest face
    // nearest_dist     (float)   : Distance to the nearest point on the mesh
    // nearest_entity   (Entity)  : Face entity (face, edge, vertex) that is
    //                              closest to the target point
    //
    // Input Arguments
    // ---------------
    // node_id  (integer)       : Current node in the tree (initially zero)
    // point    (Point)         : Target point
    // vertices (List of Point) : All vertices
    // faces    (List of Point) : All faces
    // nodes    (*Node)         : Search tree
    template <typename NearestPoint, typename Point, typename Vertices, typename Faces> 
    __host__ __device__ static inline
    void query_sqdist(
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

            auto face_id      = node.right;
            auto facevertices = Base::get_facevertices(face_id, vertices, faces);

            NearestEntity       maybe_entity;
            StaticPointScalar   maybe_point;
            scalar_t maybe_dist = Utils::sqdist_unsigned(
                maybe_entity, maybe_point, point, facevertices
            );

            if (maybe_dist < nearest_dist)
            {
                nearest_face   = face_id;
                nearest_dist   = maybe_dist;
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
                return query_sqdist(
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

    // Compute the unsigned distance from a point to the mesh
    //
    // Input Arguments
    // ---------------
    // point    (Point[f])         : Target point
    // vertices (List of Point[f]) : All vertices coordinates
    // faces    (List of Point[i]) : All faces
    // tree     (*Node)            : Search tree
    //
    // Output Arguments (optional)
    // ---------------------------
    // ptr_nearest_vertex (*integer) : Index of nearest vertex in the nearest face
    // ptr_nearest_face   (*integer) : Index of the nearest face
    // ptr_nearest_entity (*Entity)  : Descriptor of the nearest entity in the face
    //
    // Returns
    // -------
    // dist (floating) : Absolute distance to nearest point
    template <typename Point, typename Vertices, typename Faces> 
    __host__ __device__ static inline
    scalar_t unsigned_sqdist(
            const Point    & point, 
            const Vertices & vertices, 
            const Faces    & faces, 
            const Node     * tree,
            index_t        * ptr_nearest_vertex = nullptr,
            index_t        * ptr_nearest_face   = nullptr,
            NearestEntity  * ptr_nearest_entity = nullptr
    )
    {
        // Initialize output variables
        index_t             nearest_face;
        StaticPointScalar   nearest_point;
        NearestEntity       nearest_entity;
        scalar_t            nearest_dist = static_cast<scalar_t>(1./0.);

        // Find nearest face
        query_sqdist(
            nearest_face,       // out
            nearest_dist,       // out
            nearest_entity,     // out
            nearest_point,      // out
            static_cast<index_t>(0),
            point,              // inp
            vertices,           // inp
            faces,              // inp
            tree                // inp
        );

        // Save optional outputs
        if (ptr_nearest_face)   *ptr_nearest_face   = nearest_face;
        if (ptr_nearest_entity) *ptr_nearest_entity = nearest_entity;
        if (ptr_nearest_vertex)
            *ptr_nearest_vertex = Base::get_nearest_vertex(
                faces[nearest_face], nearest_point, vertices
            );

        return nearest_dist;
    }

    // Compute the signed distance from a point to the mesh
    //
    // Input Arguments
    // ---------------
    // point        (Point[f])         : Target point
    // vertices     (List of Point[f]) : All vertices coordinates
    // faces        (List of Point[i]) : All faces
    // tree         (*Node)            : Search tree
    // normfaces    (List of Point[f]) : Normals to all faces
    // normedge     (List of Point[f]) : Normals to all edges (if 3D)
    // normvertices (List of Point[f]) : Normals to all vertices
    //
    // Output Arguments (optional)
    // ---------------------------
    // ptr_nearest_vertex (*integer) : Index of nearest vertex in the nearest face
    // ptr_nearest_face   (*integer) : Index of the nearest face
    // ptr_nearest_entity (*Entity)  : Descriptor of the nearest entity in the face
    //
    // Returns
    // -------
    // dist (floating) : Signed distance to nearest point
    template <typename Point, typename Vertices, typename Faces, 
              typename NormFaces, typename NormEdges, typename NormVertices>  
    __host__ __device__ static inline
    scalar_t signed_sqdist(
            const Point        & point, 
            const Vertices     & vertices, 
            const Faces        & faces, 
            const Node         * tree, 
            const NormFaces    & normfaces, 
            const NormEdges    & normedges, 
            const NormVertices & normvertices,
            index_t            * ptr_nearest_vertex = nullptr,
            index_t            * ptr_nearest_face   = nullptr,
            NearestEntity      * ptr_nearest_entity = nullptr
    )
    {
        // Initialize output variables
        index_t             nearest_face;
        StaticPointScalar   nearest_point;
        NearestEntity       nearest_entity;
        scalar_t            nearest_dist = static_cast<scalar_t>(1./0.);

        // Find nearest face
        query_sqdist(
            nearest_face,
            nearest_dist,
            nearest_entity,
            nearest_point,
            static_cast<index_t>(0),
            point,
            vertices, 
            faces, 
            tree
        );

        // Save optional outputs
        if (ptr_nearest_face)   *ptr_nearest_face   = nearest_face;
        if (ptr_nearest_entity) *ptr_nearest_entity = nearest_entity;
        if (ptr_nearest_vertex)
            *ptr_nearest_vertex = Base::get_nearest_vertex(
                faces[nearest_face], nearest_point, vertices
            );

        // load normals into a compact array
        auto normals = Base::get_normals(
            nearest_face,
            vertices,
            faces,
            normfaces,
            normedges,
            normvertices
        );

        // compute sign from dot product <ray, normal>
        scalar_t sign = Utils::sign(
            point, 
            nearest_point, 
            normals, 
            nearest_entity
        );

        return nearest_dist * sign;
    }

    // Compute the unsigned distance from a point to the mesh, and its gradient
    //
    // Output Arguments
    // ----------------
    // nearest_face (integer)       : Index of the nearest face
    // grad_point   (Point)         : Derivative of the distance wrt the target point
    // grad_vert    (List of Point) : Derivative of the distance wrt to each vertex
    // 
    // Input Arguments
    // ---------------
    // point    (Point[f])         : Target point
    // vertices (List of Point[f]) : All vertices coordinates
    // faces    (List of Point[i]) : All faces
    // tree     (*Node)            : Search tree
    //
    // Returns
    // -------
    // dist (floating) : Absolute distance to nearest point
    template <typename Point, typename Vertices, typename Faces,
              typename GradPoint, typename GradVert> 
    __host__ __device__ static inline
    scalar_t unsigned_sqdist_grad(
            index_t        & nearest_face,
            GradPoint      & grad_point,
            GradVert       & grad_vert,
            const Point    & point, 
            const Vertices & vertices, 
            const Faces    & faces, 
            const Node     * tree
    )
    {
        // Initialize output variables
        StaticPointScalar   nearest_point;
        NearestEntity       nearest_entity;
        scalar_t            nearest_dist = static_cast<scalar_t>(1./0.);

        // find nearest face
        query_sqdist(
            nearest_face,
            nearest_dist,
            nearest_entity,
            nearest_point,
            static_cast<index_t>(0),
            point,
            vertices, 
            faces, 
            tree
        );

        // compute gradients, conditioned on the nearest face
        // (this will recompute the nearest entity and distance, which 
        // is a bit dumb...)
        nearest_dist = Utils::sqdist_unsigned_grad(
            nearest_entity, 
            nearest_point, 
            grad_vert,
            grad_point,
            point, 
            Base::get_facevertices(nearest_face, vertices, faces)
        );

        return nearest_dist;
    }

    // Compute the signed distance from a point to the mesh, and its gradient
    //
    // Output Arguments
    // ----------------
    // nearest_face (integer)       : Index of the nearest face
    // grad_point   (Point)         : Derivative of the distance wrt the target point
    // grad_vert    (List of Point) : Derivative of the distance wrt to each vertex
    // 
    // Input Arguments
    // ---------------
    // point        (Point[f])         : Target point
    // vertices     (List of Point[f]) : All vertices coordinates
    // faces        (List of Point[i]) : All faces
    // tree         (*Node)            : Search tree
    // normfaces    (List of Point[f]) : Normals to all faces
    // normedge     (List of Point[f]) : Normals to all edges (if 3D)
    // normvertices (List of Point[f]) : Normals to all vertices
    //
    // Returns
    // -------
    // dist (floating) : Signed distance to nearest point
    template <typename GradPoint, typename GradVert, 
              typename Point, typename Vertices, typename Faces, 
              typename NormFaces, typename NormEdges, typename NormVertices>  
    __host__ __device__ static inline
    scalar_t signed_sqdist_grad(
            index_t            & nearest_face,
            GradPoint          & grad_point,
            GradVert           & grad_vert,
            const Point        & point, 
            const Vertices     & vertices, 
            const Faces        & faces, 
            const Node         * tree, 
            const NormFaces    & normfaces, 
            const NormEdges    & normedges, 
            const NormVertices & normvertices
    )
    {
        // Initialize output variables
        StaticPointScalar   nearest_point;
        NearestEntity       nearest_entity;
        scalar_t            nearest_dist = static_cast<scalar_t>(1./0.);

        // find nearest face
        query_sqdist(
            nearest_face,
            nearest_dist,
            nearest_entity,
            nearest_point,
            static_cast<index_t>(0),
            point,
            vertices, 
            faces, 
            tree
        );

        // compute gradients, conditioned on the nearest face
        // (this will recompute the nearest entity and distance, which 
        // is a bit dumb...)
        nearest_dist = Utils::sqdist_unsigned_grad(
            nearest_entity, 
            nearest_point, 
            grad_vert,
            grad_point,
            point, 
            Base::get_facevertices(nearest_face, vertices, faces)
        );

        // load normals into a compact array
        auto normals = Base::get_normals(
            nearest_face,
            vertices,
            faces,
            normfaces,
            normedges,
            normvertices
        );

        // compute sign from dot product <ray, normal>
        scalar_t sign = Utils::sign(
            point, 
            nearest_point, 
            normals, 
            nearest_entity
        );

        grad_point.mul_(sign);
        for (offset_t d=0; d<D; ++d)
            grad_vert[d].mul_(sign);

        return nearest_dist * sign;
    }
};

} // namespace distance_mesh
} // namespace jf

#endif // JF_DISTANCE_MESH_CPP_H