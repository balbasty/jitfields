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
struct MeshDistUtil {};

// -----------------------------------------------------------------------------
//                              2D IMPLEMENTATION
// -----------------------------------------------------------------------------

template <typename scalar_t,  typename offset_t>
struct MeshDistUtil<2, scalar_t, offset_t> {
    static constexpr int D = 2;

    // Compute the sign of the distance function (i.e., whether the target 
    // point is inside or outside the mesh)
    //
    // Input Arguments
    // ---------------
    // point (Point2d) : Coordinates of the target point
    // nearest_point (Point2d) : Coordinates of the nearest point on the mesh
    // pseudonormals (3-List of Point2d) : Normals to the face and its ends
    // nearest_entity (NearestEntity) : Entity-type (face or vertex) of the nearest point
    //
    // Returns
    // -------
    // sign (float) : -1 if the point is inside the mesh, 1 if it is outside
    template <typename Point, typename NearestPoint, typename Normals>
    __host__ __device__ static inline
    scalar_t sign(
        const Point         & point, 
        const NearestPoint  & nearest_point, 
        const Normals       & pseudonormals,
        const NearestEntity & nearest_entity)
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
            default:
                break;
        }
        auto pseudonormal = pseudonormals[i];


        auto u = point - nearest_point;
        scalar_t s = static_cast<scalar_t>(u.dot(pseudonormal) >= 0 ? 1 : -1);
        return s;
    }

    // Compute the squared distance between a point and a segment
    //
    // Output Arguments
    // ----------------
    // nearest_entity (NearestEntity) : Entity-type (face or vertex) of the nearest point
    // nearest_point (Point2d) : Coordinates of the nearest point on the mesh
    // vertices_grad (* 2-List of Point2d) : Gradient of the squared distance wrt the vertices
    // point_grad (* Point2d) : Gradient of the squared distance wrt the target point
    // 
    // Input Arguments
    // ---------------
    // point (Point2d) : Coordinates of the target point
    // vertices (2-List of Point2d) : Endpoints of the current segment
    //
    // Returns
    // -------
    // dist (float) : Squared distance to the nearest point on the segment
    template <typename NearestPoint, typename Point, typename Vertices, typename VertGrad, typename PointGrad>
    __host__ __device__ static inline 
    scalar_t sqdist_unsigned_grad(
        NearestEntity   & nearest_entity,
        NearestPoint    & nearest_point, 
        VertGrad        & vertices_grad,
        PointGrad       & point_grad,
        const Point     & point,
        const Vertices  & vertices)
    {
        constexpr scalar_t zero = static_cast<scalar_t>(0);
        constexpr scalar_t one  = static_cast<scalar_t>(1);
        constexpr scalar_t two  = static_cast<scalar_t>(2);

        auto edge = vertices[1] - vertices[0];

        auto diff0 = point - vertices[0];
        auto dot0 = diff0.dot(edge);
        if (dot0 <= 0)
        {
            nearest_entity = NearestEntity::V0;
            nearest_point.copy_(vertices[0]);
            vertices_grad[0].copy_(diff0).mul_(-two);
            vertices_grad[1].copy_(zero);
            point_grad.copy_(diff0).mul_(two);
            return diff0.sqnorm();
        }

        auto diff1 = point - vertices[1];
        auto dot1 = diff1.dot(edge);
        if (diff1.dot(edge) >= 0)
        {
            nearest_entity = NearestEntity::V1;
            nearest_point.copy_(vertices[1]);
            vertices_grad[0].copy_(zero);
            vertices_grad[1].copy_(diff1).mul_(-two);
            point_grad.copy_(diff1).mul_(two);
            return diff1.sqnorm();
        }

        nearest_entity = NearestEntity::F;
        auto edgesqnorm = edge.sqnorm();
        auto alpha0 = dot0 / edgesqnorm;
        auto alpha1 = dot1 / edgesqnorm;
        nearest_point.addto_(vertices[0], edge, alpha0);
        vertices_grad[0].copy_(edge, -one / alpha1).add_(diff1).mul_((+two) * alpha1);
        vertices_grad[1].copy_(edge, -one / alpha0).add_(diff0).mul_((-two) * alpha0);
        point_grad.subto_(diff0, edge, alpha0).mul_(two);
        return diff0.sqnorm() - (dot0 * dot0) / edgesqnorm;
    }

    // Compute the squared distance between a point and a segment
    //
    // Output Arguments
    // ----------------
    // nearest_entity (NearestEntity) : Entity-type (face or vertex) of the nearest point
    // nearest_point (Point2d) : Coordinates of the nearest point on the mesh
    //
    // Input Arguments
    // ---------------
    // point (Point2d) : Coordinates of the target point
    // vertices (2-List of Point2d) : Endpoints of the current segment
    //
    // Returns
    // -------
    // dist (float) : Squared distance to the nearest point on the segment
    template <typename NearestPoint, typename Point, typename Vertices>
    __host__ __device__ static inline 
    scalar_t sqdist_unsigned(
        NearestEntity   & nearest_entity, 
        NearestPoint    & nearest_point, 
        const Point     & point,
        const Vertices  & vertices)
    {
        auto edge = vertices[1] - vertices[0];

        auto diff0 = point - vertices[0];
        auto dot0 = diff0.dot(edge);
        if (dot0 <= 0)
        {
            nearest_entity = NearestEntity::V0;
            nearest_point.copy_(vertices[0]);
            return diff0.sqnorm();
        }

        auto diff1 = point - vertices[1];
        if (diff1.dot(edge) >= 0)
        {
            nearest_entity = NearestEntity::V1;
            nearest_point.copy_(vertices[1]);
            return diff1.sqnorm();
        }

        nearest_entity = NearestEntity::F;
        auto edgesqnorm = edge.sqnorm();
        nearest_point.addto_(vertices[0], edge, dot0 / edgesqnorm);
        return diff0.sqnorm() - (dot0 * dot0) / edgesqnorm;
    }
};

// -----------------------------------------------------------------------------
//                              3D IMPLEMENTATION
// -----------------------------------------------------------------------------

template <typename scalar_t,  typename offset_t>
struct MeshDistUtil<3, scalar_t, offset_t> {
    static constexpr int D = 3;

    // Compute the sign of the distance function (i.e., whether the target 
    // point is inside or outside the mesh)
    //
    // Input Arguments
    // ---------------
    // point (Point3d) : Coordinates of the target point
    // nearest_point (Point3d) : Coordinates of the nearest point on the mesh
    // pseudonormals (7-List of Point2d) : Normals to the face, vertices, and edges
    // nearest_entity (NearestEntity) : Entity-type (face, edge or vertex) of the nearest point
    //
    // Returns
    // -------
    // sign (float) : -1 if the point is inside the mesh, 1 if it is outside
    template <typename Point, typename NearestPoint, typename Normals>
    __host__ __device__ static inline
    scalar_t sign(
        const Point         & point,
        const NearestPoint  & nearest_point, 
        const Normals       & pseudonormals,
        const NearestEntity & nearest_entity)
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

    // Compute the squared distance between a point and a segment
    //
    // Output Arguments
    // ----------------
    // nearest_entity (NearestEntity) : Entity-type (face, edge or vertex) of the nearest point
    // nearest_point (Point3d) : Coordinates of the nearest point on the mesh
    // vertices_grad (* 3-List of Point3d) : Gradient of the squared distance wrt the vertices
    // point_grad (* Point3d) : Gradient of the squared distance wrt the target point
    //
    // Input Arguments
    // ---------------
    // point (Point3d) : Coordinates of the target point
    // vertices (3-List of Point3d) : Vertices of the current triangle
    //
    // Returns
    // -------
    // dist (float) : Squared distance to the nearest point on the triangle
    template <typename NearestPoint, typename Point, typename Vertices, typename VertGrad, typename PointGrad>
    __host__ __device__ static inline 
    scalar_t sqdist_unsigned_grad(
        NearestEntity   & nearest_entity,
        NearestPoint    & nearest_point, 
        VertGrad        & vertices_grad,
        PointGrad       & point_grad,
        const Point     & point,
        const Vertices  & vertices)
    {
        constexpr scalar_t zero = static_cast<scalar_t>(0);
        constexpr scalar_t one  = static_cast<scalar_t>(1);
        constexpr scalar_t two  = static_cast<scalar_t>(2);

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

        vertices_grad[0].copy_(zero);
        vertices_grad[1].copy_(zero);
        vertices_grad[2].copy_(zero);

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
                            s = one;
                            d2 = a00 + 2 * b0 + c;
                            point_grad.subto_(point, vertices[1]).mul_(two);
                            vertices_grad[1].copy_(point_grad, -one);
                        }
                        else
                        {
                            nearest_entity = NearestEntity::E01;
                            s = -b0 / a00;
                            d2 = b0 * s + c;
                            point_grad.subto_(diff, edge0, -s).mul_(-two);
                            vertices_grad[0].copy_(point_grad, s - one);
                            vertices_grad[1].copy_(point_grad, -s);
                        }
                    }
                    else
                    {
                        s = zero;
                        if (b1 >= 0)
                        {
                            nearest_entity = NearestEntity::V0;
                            t = zero;
                            d2 = c;
                            point_grad.copy_(diff).mul_(-two);
                            vertices_grad[0].copy_(diff).mul_(two);
                        }
                        else if (-b1 >= a11)
                        {
                            nearest_entity = NearestEntity::V2;
                            t = one;
                            d2 = a11 + 2 * b1 + c;
                            point_grad.subto_(point, vertices[2]).mul_(two);
                            vertices_grad[2].copy_(point_grad, -one);
                        }
                        else
                        {
                            nearest_entity = NearestEntity::E02;
                            t = -b1 / a11;
                            d2 = b1 * t + c;
                            point_grad.subto_(diff, edge1, -t).mul_(-two);
                            vertices_grad[0].copy_(point_grad, t - one);
                            vertices_grad[2].copy_(point_grad, -t);
                        }
                    }
                }
                else
                {
                    s = zero;
                    if (b1 >= 0)
                    {
                        nearest_entity = NearestEntity::V0;
                        t = zero;
                        d2 = c;
                        point_grad.copy_(diff).mul_(-two);
                        vertices_grad[0].copy_(diff).mul_(two);
                    }
                    else if (-b1 >= a11)
                    {
                        nearest_entity = NearestEntity::V2;
                        t = one;
                        d2 = a11 + 2 * b1 + c;
                        point_grad.subto_(point, vertices[2]).mul_(-two);
                        vertices_grad[2].copy_(point_grad, -one);
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E02;
                        t = -b1 / a11;
                        d2 = b1 * t + c;
                        point_grad.subto_(diff, edge1, -t).mul_(-two);
                        vertices_grad[0].copy_(point_grad, t - one);
                        vertices_grad[2].copy_(point_grad, -t);
                    }
                }
            }
            else if (t < 0)
            {
                t = zero;
                if (b0 >= 0)
                {
                    nearest_entity = NearestEntity::V0;
                    s = zero;
                    d2 = c;
                    point_grad.copy_(diff).mul_(-two);
                    vertices_grad[0].copy_(diff).mul_(two);
                }
                else if (-b0 >= a00)
                {
                    nearest_entity = NearestEntity::V1;
                    s = one;
                    d2 = a00 + 2 * b0 + c;
                    point_grad.subto_(point, vertices[1]).mul_(two);
                    vertices_grad[1].copy_(point_grad, -one);
                }
                else
                {
                    nearest_entity = NearestEntity::E01;
                    s = -b0 / a00;
                    d2 = b0 * s + c;
                    point_grad.subto_(diff, edge0, -s).mul_(-two);
                    vertices_grad[0].copy_(point_grad, s - one);
                    vertices_grad[1].copy_(point_grad, -s);
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
                auto n = edge1.cross(edge0);
                auto ndot = diff.dot(n);
                point_grad.copy_(n, - 2 * diff.dot(n) * invDet);
                {
                    auto edge2 = vertices[2] - vertices[1];
                    auto tmp = point_grad + 2 * (diff + edge0);
                    vertices_grad[0].crossto_(tmp, edge2).mul_(-invDet * (ndot + edge0.dot(n)));
                }
                {
                    auto tmp = point_grad + 2 * diff;
                    vertices_grad[1].crossto_(tmp, edge1).mul_( invDet * ndot);
                    vertices_grad[2].crossto_(tmp, edge0).mul_(-invDet * ndot);
                }
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
                        s = one;
                        t = zero;
                        d2 = a00 + 2 * b0 + c;
                        point_grad.subto_(point, vertices[1]).mul_(two);
                        vertices_grad[1].copy_(point_grad, -one);
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E12;
                        s = numer / denom;
                        t = 1 - s;
                        d2 = s * (a00 * s + a01 * t + 2 * b0) +
                             t * (a01 * s + a11 * t + 2 * b1) + c;
                        auto b = ((a01 - a00) + (b1 - b0)) / (a00 + a11 - 2*a01);
                        point_grad.subto_(diff + edge0, edge1 - edge0, b).mul_(-two);
                        vertices_grad[1].copy_(point_grad, -(b + one));
                        vertices_grad[2].copy_(point_grad, b);
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
                        point_grad.subto_(point, vertices[2]).mul_(two);
                        vertices_grad[2].copy_(point_grad, -one);
                    }
                    else if (b1 >= 0)
                    {
                        nearest_entity = NearestEntity::V0;
                        t = static_cast<scalar_t>(0);
                        d2 = c;
                        point_grad.copy_(diff).mul_(-two);
                        vertices_grad[0].copy_(diff).mul_(two);
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E02;
                        t = -b1 / a11;
                        d2 = b1 * t + c;
                        point_grad.subto_(diff, edge1, -t).mul_(-two);
                        vertices_grad[0].copy_(point_grad, t - one);
                        vertices_grad[2].copy_(point_grad, -t);
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
                        t = one;
                        s = zero;
                        d2 = a11 + 2 * b1 + c;
                        point_grad.subto_(point, vertices[2]).mul_(two);
                        vertices_grad[2].copy_(point_grad, -one);
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E12;
                        t = numer / denom;
                        s = one - t;
                        d2 = s * (a00 * s + a01 * t + 2 * b0) +
                             t * (a01 * s + a11 * t + 2 * b1) + c;
                        auto b = ((a01 - a00) + (b1 - b0)) / (a00 + a11 - 2*a01);
                        point_grad.subto_(diff + edge0, edge1 - edge0, b).mul_(-two);
                        vertices_grad[1].copy_(point_grad, -(b + one));
                        vertices_grad[2].copy_(point_grad, b);
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
                        point_grad.subto_(point, vertices[1]).mul_(two);
                        vertices_grad[1].copy_(point_grad, -one);
                    }
                    else if (b0 >= 0)
                    {
                        nearest_entity = NearestEntity::V0;
                        s = static_cast<scalar_t>(0);
                        d2 = c;
                        point_grad.copy_(diff).mul_(-two);
                        vertices_grad[0].copy_(diff).mul_(two);
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E01;
                        s = -b0 / a00;
                        d2 = b0 * s + c;
                        point_grad.subto_(diff, edge0, -s).mul_(-two);
                        vertices_grad[0].copy_(point_grad, s - one);
                        vertices_grad[1].copy_(point_grad, -s);
                    }
                }
            }
            else
            {
                numer = a11 + b1 - a01 - b0;
                if (numer <= 0)
                {
                    nearest_entity = NearestEntity::V2;
                    s = zero;
                    t = one;
                    d2 = a11 + 2 * b1 + c;
                    point_grad.subto_(point, vertices[2]).mul_(two);
                    vertices_grad[2].copy_(point_grad, -one);
                }
                else
                {
                    denom = a00 - 2 * a01 + a11;
                    if (numer >= denom)
                    {
                        nearest_entity = NearestEntity::V1;
                        s = one;
                        t = zero;
                        d2 = a00 + 2 * b0 + c;
                        point_grad.subto_(point, vertices[1]).mul_(two);
                        vertices_grad[1].copy_(point_grad, -one);
                    }
                    else
                    {
                        nearest_entity = NearestEntity::E12;
                        s = numer / denom;
                        t = one - s;
                        d2 = s * (a00 * s + a01 * t + 2 * b0) +
                             t * (a01 * s + a11 * t + 2 * b1) + c;
                        auto b = ((a01 - a00) + (b1 - b0)) / (a00 + a11 - 2*a01);
                        point_grad.subto_(diff + edge0, edge1 - edge0, b).mul_(-two);
                        vertices_grad[1].copy_(point_grad, -(b + one));
                        vertices_grad[2].copy_(point_grad, b);
                    }
                }
            }
        }

        for (int d=0; d<3; ++d)
            nearest_point[d] = vertices[0][d] + s * edge0[d] + t * edge1[d];

        // Account for numerical round-off error.
        if (d2 < 0)
            d2 = static_cast<scalar_t>(0);
        return d2;
    }

    // Compute the squared distance between a point and a segment
    //
    // Output Arguments
    // ----------------
    // nearest_entity (NearestEntity) : Entity-type (face, edge or vertex) of the nearest point
    // nearest_point (Point3d) : Coordinates of the nearest point on the mesh
    //
    // Input Arguments
    // ---------------
    // point (Point3d) : Coordinates of the target point
    // vertices (3-List of Point3d) : Vertices of the current triangle
    //
    // Returns
    // -------
    // dist (float) : Squared distance to the nearest point on the triangle
    template <typename NearestPoint, typename Point, typename Vertices>
    __host__ __device__ static inline 
    scalar_t sqdist_unsigned(
        NearestEntity   & nearest_entity,
        NearestPoint    & nearest_point,
        const Point     & point,
        const Vertices  & vertices)
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
        if (d2 < 0)
            d2 = static_cast<scalar_t>(0);
        return d2;
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

    // A sphere encoded by its center and radius
    struct BoundingSphere {
        virtual ~BoundingSphere() {}
        StaticPointScalar   center;
        scalar_t            radius;
    };

    // A node in the search three
    struct Node {
        virtual ~Node() {}
        BoundingSphere  bv_left;      // Bounding sphere of the left child
        BoundingSphere  bv_right;     // Bounding sphere of the right child
        index_t         left   = -1;  // Index of the left child node
        index_t         right  = -1;  // Index of the right child node
        index_t         parent = -1;  // Index of the parent node
    };

    // Compute the bounding sphere of a face
    //
    // Input Arguments
    // ---------------
    // face (List of Point[f]) : Coordinates of the vertices of a face
    //
    // Returns
    // -------
    // sphere (BoundingSphere) : A sphere that bounds the vertices
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

    // Return the coordinates of the vertices of a face
    //
    // Input Arguments
    // ---------------
    // face_id  (integer)          : Face index
    // vertices (List of Point[f]) : Coordinates of all vertices
    // faces    (List of Point[i]) : All faces (each face = indices of its vertices)
    // 
    // Returns
    // -------
    // facevertices (List of Point[f]) : Vertex coordinates of `face_id`.
    template <typename Vertices, typename Faces>
    __host__ __device__ static inline
    StaticPointList<D, D, scalar_t> get_facevertices(
        index_t          face_id, 
        const Vertices & vertices, 
        const Faces    & faces
    )
    {
        auto face = faces[face_id];
        auto facevertices = StaticPointList<D, D, scalar_t>();
        for (offset_t d=0; d<D; ++d)
            facevertices[d].copy_(vertices[face[d]]);
        return facevertices;
    }

    // Returns the normals to all entities of a face
    //
    // Input Arguments
    // ---------------
    // face_id      (integer)          : Face index
    // vertices     (List of Point[f]) : Coordinates of all vertices
    // faces        (List of Point[i]) : All faces (each face = indices of its vertices)
    // normfaces    (List of Point[f]) : Normals to all faces
    // normedges    (List of Point[f]) : Normals to all edges (if 3D)
    // normvertices (List of Point[f]) : Normals to all vertices
    // 
    // Returns
    // -------
    // facevertices (List of Point[f]) : Normals to a face
    //    - 2D: ordered as F, V0, V1
    //    - 3D: ordered as F, E01, E02, E12, V0, V1, V2
    template <typename Vertices, typename Faces, 
              typename NormFaces, typename NormEdges, typename NormVertices>
    __host__ __device__ static inline
    StaticPointList<D+1+(D == 3 ? D : 0), D, scalar_t>
    get_normals(
        index_t              face_id, 
        const Vertices     & vertices, 
        const Faces        & faces,
        const NormFaces    & normfaces, 
        const NormEdges    & normedges, 
        const NormVertices & normvertices
    )
    {
        auto normals = StaticPointList<D+1+(D == 3 ? D : 0), D, scalar_t>();
        auto face    = faces[face_id];
        normals[0].copy_(normfaces[face_id]);
        if (D == 3)
        {
            auto normedge = normedges[face_id];
            for (offset_t d=0; d<D; ++d)
            {
                normals[1+d].copy_(normvertices[face[d]]);
                normals[1+D+d].copy_(normedge[d]);
            }
        }
        else
        {
            for (offset_t d=0; d<D; ++d)
                normals[1+d].copy_(normvertices[face[d]]);
        }

        return normals;
    }

    // Returns the vertex nearest to a point in a face
    //
    // Input Arguments
    // ---------------
    // face      (Point[i])         : Indices of the vertices of a face
    // point     (Point[f])         : Target point
    // vertices  (List of Point[f]) : Coordinates of all vertices
    //
    // Returns
    // -------
    // vertex_index (integer) : Index of the nearest vertex
    template <typename Point, typename Vertices, typename Face>
    __host__ __device__ static inline
    index_t get_nearest_vertex(
        const Face     & face,
        const Point    & point, 
        const Vertices & vertices)
    {
        scalar_t best_dist   = static_cast<scalar_t>(1./0.);
        index_t  best_vertex = 0;
        for (offset_t d=0; d<D; ++d)
        {
            auto vertex_id = face[d];
            auto dist = (vertices[vertex_id]-point).norm();
            if (dist < best_dist)
            {
                best_dist   = dist;
                best_vertex = vertex_id;
            }
        }
        return best_vertex;
    }

    template <typename NearestPoint, typename Point, typename Vertices, typename Faces> 
    __host__ __device__ static inline
    scalar_t _unsigned_sqdist_naive(
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
            }

            if (maybe_dist < nearest_dist)
            {
                nearest_face   = n;
                nearest_dist   = maybe_dist;
                nearest_entity = maybe_entity;
                nearest_point  = maybe_point;
            }
        }

        return nearest_dist;
    }

    template <typename Point, typename Vertices, typename Faces> 
    __host__ __device__ static inline
    scalar_t unsigned_sqdist_naive(
            const Point    & point, 
            const Vertices & vertices, 
            const Faces    & faces,
            index_t * nearest_vertex = nullptr
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

        // get index of vertex nearest to the projection
        if (nearest_vertex)
            *nearest_vertex = get_nearest_vertex(faces[nearest_face], nearest_point, vertices);
    }

    template <typename Point, typename Vertices, typename Faces, 
              typename NormFaces, typename NormEdges, typename NormVertices> 
    __host__ __device__ static inline
    scalar_t signed_sqdist_naive(
            const Point        & point, 
            const Vertices     & vertices, 
            const Faces        & faces, 
            const NormFaces    & normfaces, 
            const NormEdges    & normedges, 
            const NormVertices & normvertices,
                  index_t      * nearest_vertex = nullptr
    )
    {
        index_t             nearest_face;
        StaticPointScalar   nearest_point;
        NearestEntity       nearest_entity;

        // compute unsigned distance and return index of nearest triangle
        scalar_t dist = _unsigned_sqdist_naive(
            nearest_face,
            nearest_entity,
            nearest_point,
            point,
            vertices, 
            faces
        );

        // get index of vertex nearest to the projection
        if (nearest_vertex)
            *nearest_vertex = get_nearest_vertex(faces[nearest_face], nearest_point, vertices);

        // load normals into a compact array
        auto normals     = StaticPointList<D+1+(D == 3 ? D : 0), D, scalar_t>();
        auto face        = faces[nearest_face];
        normals[0].copy_(normfaces[nearest_face]);
        if (D == 3)
        {
            auto normedge = normedges[nearest_face];
            for (offset_t d=0; d<D; ++d)
            {
                normals[1+d].copy_(normvertices[face[d]]);
                normals[1+D+d].copy_(normedge[d]);
            }
        }
        else
        {
            for (offset_t d=0; d<D; ++d)
                normals[1+d].copy_(normvertices[face[d]]);
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