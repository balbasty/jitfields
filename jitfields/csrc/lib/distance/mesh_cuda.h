#ifndef JF_DISTANCE_MESH_CUDA_H
#define JF_DISTANCE_MESH_CUDA_H
#include "../utils.h"
#include "mesh_utils.h"
#include "mesh.h"


namespace jf {
namespace distance_mesh {


// =============================================================================
//
//                                  GENERIC API
//
// =============================================================================


template <int D, typename scalar_t, typename index_t, typename offset_t>
struct MeshDistCuda: public MeshDist<D, scalar_t, index_t, offset_t> 
{
    using Base              = MeshDist<D, scalar_t, index_t, offset_t>;
    using Utils             = MeshDistUtil<D, scalar_t, offset_t>;
    using StaticPointScalar = StaticPoint<D, scalar_t>;
    using BoundingSphere    = typename Base::BoundingSphere;
    using Node              = typename Base::Node;


    // Find the face nearest to a target point
    //
    // we can't use recursions in cuda (because stack size must be known 
    // at compile time) so we must unroll the recursion, which is a pain. 
    // This works though!
    //
    // Output Arguments
    // ----------------
    // nearest_face     (integer) : Index of the nearest face
    // nearest_dist     (float)   : Distance to the nearest point on the mesh
    // nearest_entity   (Entity)  : Face entity (face, edge, vertex) that is
    //                              closest to the target point
    // nearest_point    (Point)   : Coordinates of the nearest point
    //
    // Input/Output Arguments
    // ----------------------
    // trace (*char) : A buffer that is used to navigate the tree.
    //
    // Input Arguments
    // ---------------
    // point    (Point)         : Target point
    // vertices (List of Point) : All vertices
    // faces    (List of Point) : All faces
    // nodes    (*Node)         : Search tree
    template <typename NearestPoint, typename Point, typename Vertices, typename Faces, typename Trace> 
    __host__ __device__ static inline
    void query_sqdist(
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
        index_t level   = 0;

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

                if (maybe_dist < nearest_dist)
                {
                    nearest_face   = face_id;
                    nearest_dist   = maybe_dist;
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

    // Compute the unsigned distance from a point to the mesh
    //
    // Input Arguments
    // ---------------
    // point     (Point[f])         : Target point
    // vertices  (List of Point[f]) : All vertices coordinates
    // faces     (List of Point[i]) : All faces
    // tree      (*Node)            : Search tree
    //
    // Buffer
    // ------
    // treetrace (*uint8) : A buffer that is used to navigate the tree
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
    template <typename Point, typename Vertices, typename Faces, typename Trace> 
    __host__ __device__ static inline
    scalar_t unsigned_sqdist(
            const Point    & point, 
            const Vertices & vertices, 
            const Faces    & faces, 
            const Node     * tree,
            Trace          & treetrace,
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

        // find nearest face
        query_sqdist(
            nearest_face,
            nearest_dist,
            nearest_entity,
            nearest_point,
            point, 
            vertices, 
            faces, 
            tree,
            treetrace
        );
        if (ptr_nearest_face)   *ptr_nearest_face   = nearest_face;
        if (ptr_nearest_entity) *ptr_nearest_entity = nearest_entity;

        // get index of vertex nearest to the projection
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
    // Buffer
    // ------
    // treetrace (*uint8) : A buffer that is used to navigate the tree
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
              typename NormFaces, typename NormEdges, typename NormVertices,
              typename Trace>  
    __host__ __device__ static inline
    scalar_t signed_sqdist(
            const Point        & point, 
            const Vertices     & vertices, 
            const Faces        & faces, 
            const Node         * tree, 
            Trace              & treetrace,
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

        // find nearest face
        query_sqdist(
            nearest_face,
            nearest_dist,
            nearest_entity,
            nearest_point,
            point, 
            vertices, 
            faces, 
            tree,
            treetrace
        );
        if (ptr_nearest_face)   *ptr_nearest_face   = nearest_face;
        if (ptr_nearest_entity) *ptr_nearest_entity = nearest_entity;

        // get index of vertex nearest to the projection
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
    // Buffer
    // ------
    // treetrace (*uint8) : A buffer that is used to navigate the tree
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
              typename GradPoint, typename GradVert, typename Trace> 
    __host__ __device__ static inline
    scalar_t unsigned_sqdist_grad(
            index_t        & nearest_face,
            GradPoint      & grad_point,
            GradVert       & grad_vert,
            const Point    & point, 
            const Vertices & vertices, 
            const Faces    & faces, 
            const Node     * tree,
            Trace          & treetrace
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
            point, 
            vertices, 
            faces, 
            tree,
            treetrace
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
    // Buffer
    // ------
    // treetrace (*uint8) : A buffer that is used to navigate the tree
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
    template <typename Point, typename Vertices, typename Faces, typename Trace,
              typename NormFaces, typename NormEdges, typename NormVertices,
              typename GradPoint, typename GradVert>
    __host__ __device__ static inline
    scalar_t signed_sqdist_grad(
            index_t            & nearest_face,
            GradPoint          & grad_point,
            GradVert           & grad_vert,
            const Point        & point, 
            const Vertices     & vertices, 
            const Faces        & faces, 
            const Node         * tree, 
            Trace              & treetrace,
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
            point, 
            vertices, 
            faces, 
            tree,
            treetrace
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

#endif // JF_DISTANCE_MESH_CUDA_H