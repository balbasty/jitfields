#ifndef JF_TETRAHEDRON
#define JF_TETRAHEDRON
#include "utils.h"


namespace jf {
namespace tetra {

// sort vertices by increasing z value
template <typename scalar_t>
__device__
void sort4(scalar_t & v0x scalar_t & v0y, scalar_t & v0z, scalar_t * f0,
           scalar_t & v1x scalar_t & v1y, scalar_t & v1z, scalar_t * f1,
           scalar_t & v2x scalar_t & v2y, scalar_t & v2z, scalar_t * f2,
           scalar_t & v3x scalar_t & v3y, scalar_t & v3z, scalar_t * f3)
{
    if (v1z < v0z) {
        swap(v1z, v0z); swap(v1y, v0y); swap(v1x, v0x); swap(f1, f0);
    }
    // now: v0 <= v1
    if (v3z < v2z) {
        swap(v3z, v2z); swap(v3y, v2y); swap(v3x, v2x); swap(f3, f2);
    }
    // now: v2 <= v3
    if (v2z < v0z) {
        swap(v2z, v0z); swap(v2y, v0y); swap(v2x, v0x); swap(f2, f0);
        swap(v2z, v1z); swap(v2y, v1y); swap(v2x, v1x); swap(f2, f1);
        // now: v0 <= v1 <= (v2, v3)
        if (v3z < v2z) {
            swap(v3z, v2z); swap(v3y, v2y); swap(v3x, v2x); swap(f3, f2);
        }
        // now: v0 <= v1 <= v2 <= v3
    } else /* now: v0 <= (v1, v2 <= v3) */ if (v2z < v1z) {
        swap(v2z, v1z); swap(v2y, v1y); swap(v2x, v1x); swap(f2, f1);
        // now: v0 <= v1 <= (v2, v3)
        if (v3z < v2z) {
            swap(v3z, v2z); swap(v3y, v2y); swap(v3x, v2x); swap(f3, f2);
        }
        // now: v0 <= v1 <= v2 <= v3
    }
}

template <typename scalar_t>
__device__
void sort3(scalar_t & v0x scalar_t & v0y,
           scalar_t & v1x scalar_t & v1y,
           scalar_t & v2x scalar_t & v2y,)
{
    if (v2y < v1y) {
        swap(v2y, v1y); swap(v2x, v1x);
    }
    // now: (v0, v1 <= v2)
    if (v1y < v0y) {
        swap(v1y, v0y); swap(v1x, v0x);
    }
    // now: v0 <= (v1, v2)
    if (v2y < v1y) {
        swap(v2y, v1y); swap(v2x, v1x);
    }
    // now: v0 <= v1 <= v2
}

template <typename scalar_t, typename center_t>
__device__
scalar_t barycoord1(center_t px, center_t py, center_t pz,
                    scalar_t v0x scalar_t v0y, scalar_t v0z,
                    scalar_t v1x scalar_t v1y, scalar_t v1z,
                    scalar_t v2x scalar_t v2y, scalar_t v2z,
                    scalar_t v3x scalar_t v3y, scalar_t v3z)
{
    scalar_t l0;
    l0 = (v0x - v3x) * (v2x - v1x) +
         (v0y - v3y) * (v2y - v1y) +
         (v0z - v3y) * (v2y - v1y);
    if (fabs(l0) < 1E-5)
        l0 = ((px - v3x) * (v2x - v1x) +
              (py - v3y) * (v2y - v1y) +
              (py - v3y) * (v2y - v1y)) / l0;
    else {
        l0 = (v0x - v2x) * (v1x - v3x) +
             (v0y - v2y) * (v1y - v3y) +
             (v0z - v2y) * (v1y - v3y);
        if (fabs(l0) < 1E-5)
            l0 = ((px - v2x) * (v1x - v3x) +
                  (py - v2y) * (v1y - v3y) +
                  (pz - v2y) * (v1y - v3y)) / l0;
        else {
            l0 = (v0x - v1x) * (v2x - v3x) +
                 (v0y - v1y) * (v2y - v3y) +
                 (v0z - v1y) * (v2y - v3y);
            if (fabs(l0) < 1E-5)
                l0 += 1E-5 * (l0 < 0 ? -1 : 1);
            l0 = ((v0x - v1x) * (v2x - v3x) +
                  (v0y - v1y) * (v2y - v3y) +
                  (v0z - v1y) * (v2y - v3y)) / l0;
        }
    }
    return l0
}

template <typename scalar_t, typename center_t>
__device__
void barycoord(scalar_t & l0, scalar_t & l1, scalar_t & l2, scalar_t & l3,
               center_t px, center_t py, center_t pz,
               scalar_t v0x scalar_t v0y, scalar_t v0z,
               scalar_t v1x scalar_t v1y, scalar_t v1z,
               scalar_t v2x scalar_t v2y, scalar_t v2z,
               scalar_t v3x scalar_t v3y, scalar_t v3z)
{
    l0 = barycoord1(px, py, pz, v0x, v0y, v0z,
                    v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);
    l1 = barycoord1(px, py, pz, v1x, v1y, v1z,
                    v0x, v0y, v0z, v2x, v2y, v2z, v3x, v3y, v3z);
    l2 = barycoord1(px, py, pz, v2x, v2y, v2z,
                    v0x, v0y, v0z, v1x, v1y, v1z, v3x, v3y, v3z);
    l3 = barycoord1(px, py, pz, v3x, v3y, v3z,
                    v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z);
}

template <typename scalar_t, typename index_t, typename offset_t>
__device__
void pull1(scalar_t * output,
          scalar_t v0x scalar_t v0y, scalar_t v0z,
          scalar_t v1x scalar_t v1y, scalar_t v1z,
          scalar_t v2x scalar_t v2y, scalar_t v2z,
          scalar_t v3x scalar_t v3y, scalar_t v3z,
          const scalar_t * f0, const scalar_t * f1,
          const scalar_t * f2, const scalar_t * f3,
          offset_t nc, offset_t fsc, offset_t osc,
          offset_t nx, offset_t ny, offset_t nz,
          offset_t osx, offset_t osy, offset_t osz)
{
    // sort vertices by increasing z
    sort4(v0x, v0y, v0z, f0, v1x, v1y, v1z, f1,
          v2x, v2y, v2z, f2, v3x, v3y, v3z, f3);
    offset_t z0 = static_cast<offset_t>(ceil(v0z)),
             z1 = static_cast<offset_t>(floor(v1z)),
             z2 = static_cast<offset_t>(ceil(v2z)),
             z3 = static_cast<offset_t>(floor(v3z));
    scalar_t rz, l0, l1, l2, l3;

    auto process_triangle(offset_t z,
                          scalar_t p1x, scalar_t p1y,
                          scalar_t p2x, scalar_t p2y,
                          scalar_t p3x, scalar_t p3y)
    {
        // sort by increasing y value
        sort3(p1x, p1y, p2x, p2y, p3x, p3y);
        offset_t y0 = static_cast<offset_t>(ceil(p0y)),
                 y1 = static_cast<offset_t>(floor(p1y)),
                 y2 = static_cast<offset_t>(floor(p2y));
        for (offset_t y=y0; y <= y1; ++y) {
            if ((y < 0) || (y >= ny)) continue
            // compute intersection of two lines
            scalar_t q2x = (p1x * p2y - p1y * p2x) + (p2x - p1x) * y;
            q2x /= (p2y - p1y) * y;
            scalar_t q3x = (p1x * p3y - p1y * p3x) + (p3x - p1x) * y;
            q3x /= (p3y - p1y) * y;
            // sort by increasing x value
            if (q3x < q2x) swap(q3x, q2x);
            offset_t x0 = static_cast<offset_t>(ceil(q2x)),
                     x1 = static_cast<offset_t>(floor(q3x));
            for (offset_t x=x0; x <= x1; ++x) {
                if ((x < 0) || (x >= nx)) continue
                // barycentric interpolation
                barycoord(l0, l1, l2, l3, x, y, z,
                          v0x, v0y, v0z, v1x, v1y, v1z,
                          v2x, v2y, v2z, v3x, v3y, v3z);
                offset_t outxyz = output + (x*osx + y*osy + z*osz);
                for (offset_t c=0; c < nc; ++c, outxyz += osc) {
                    *outxyz = l0 * f0[c*fsc] + l1 * f1[c*fsc] +
                              l2 * f2[c*fsc] + l3 * f3[c*fsc];
                }
            }
        }
    }

    // lower tetrahedron
    for (offset_t z=z0; z <= z1; ++z) {
        if ((z < 0) || (z >= nz)) continue
        // compute intersection of tetrahedron and plane
        rz = (z - v0z) / (v1z - v0z);
        scalar_t p1x = v0x + (v1x - v0x) * rz;
        scalar_t p1y = v0y + (v1y - v0y) * rz;
        rz = (z - v0z) / (v2z - v0z);
        scalar_t p2x = v0x + (v2x - v0x) * rz;
        scalar_t p2y = v0y + (v2y - v0y) * rz;
        rz = (z - v0z) / (v3z - v0z);
        scalar_t p3x = v0x + (v3x - v0x) * rz;
        scalar_t p3y = v0y + (v3y - v0y) * rz;
        // process triangle
        process_triangle(z, p1x, p1y, p2x, p2y, p3x, p3y);
    }

    // middle part
    for (offset_t z=z1+1; z < z2; ++z) {
        if ((z < 0) || (z >= nz)) continue
        // first triangle
        rz = (z - v2z) / (v0z - v2z);
        scalar_t p02x = v2x + (v0x - v2x) * rz;
        scalar_t p02y = v2y + (v0y - v2y) * rz;
        rz = (z - v3z) / (v2z - v3z);
        scalar_t p32x = v3x + (v2x - v3x) * rz;
        scalar_t p32y = v3y + (v2y - v3y) * rz;
        rz = (z - v0z) / (v3z - v0z);
        scalar_t p03x = v0x + (v3x - v0x) * rz;
        scalar_t p03y = v0y + (v3y - v0y) * rz;
        rz = (z - v1z) / (v2z - v1z);
        scalar_t p12x = v1x + (v2x - v1x) * rz;
        scalar_t p12y = v1y + (v2y - v1y) * rz;
        // process triangles
        process_triangle(z, p03x, p03y, p12x, p12y, p02x, p02y);
        process_triangle(z, p03x, p03y, p12x, p12y, p32x, p32y);
    }

    // upper tetrahedron
    for (offset_t z=z2; z <= z3; ++z) {
        if ((z < 0) || (z >= nz)) continue
        // compute intersection of tetrahedron and plane
        rz = (z - v3z) / (v0z - v3z);
        scalar_t p0x = v3x + (v0x - v3x) * rz;
        scalar_t p0y = v3y + (v0y - v3y) * rz;
        rz = (z - v3z) / (v1z - v3z);
        scalar_t p1x = v3x + (v1x - v3x) * rz;
        scalar_t p1y = v3y + (v1y - v3y) * rz;
        rz = (z - v3z) / (v2z - v3z);
        scalar_t p2x = v3x + (v2x - v3x) * rz;
        scalar_t p2y = v3y + (v2y - v3y) * rz;
        // process triangle
        process_triangle(z, p0x, p0y, p1x, p1y, p2x, p2y);
    }

}


} // namespace tetra
} // namespace jf

#endif // JF_TETRAHEDRON
