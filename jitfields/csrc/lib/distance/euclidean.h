// "Distance Transforms of Sampled Functions"
// Pedro F. Felzenszwalb & Daniel P. Huttenlocher
// Theory of Computing (2012)
// https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
//
// This algorithm works by upper-bounding the Euclidean distance with
// the lower envelope of a series of parabolas.
#ifndef JF_DISTANCE_E
#define JF_DISTANCE_E
#include "../cuda_switch.h"
#include "../utils.h"

namespace jf {
namespace distance_e {

// This may be needed when working with half precision?
// (I can't remember, but it's probably here for a reason)
template <typename out_t, typename in_t>
__device__ inline
out_t mycast(in_t x)
{
    return static_cast<out_t>(static_cast<float>(x));
}


// Compute the intersection point between two parabolas
template <typename offset_t, typename scalar_t>
__device__
scalar_t intersection(scalar_t * f, offset_t * v, scalar_t w2,
                      offset_t k, offset_t q,
                      offset_t size, offset_t stride_buf)
{
    offset_t vk = v[k * stride_buf];
    scalar_t fvk = f[vk * stride_buf];
    scalar_t fq = f[q * stride_buf];
    offset_t a = q - vk, b = q + vk;
    scalar_t s = fq - mycast<scalar_t>(fvk);
    s += w2 * mycast<scalar_t>(a * b);
    s /= mycast<scalar_t>(2) * w2 * mycast<scalar_t>(a);
    return s;
}

// Compute the squared distance in each voxel based on the location of
// the parabolas
template <typename offset_t, typename scalar_t>
__device__
void fillin(scalar_t * f, offset_t * v, scalar_t * z, scalar_t * d, scalar_t w2,
            offset_t size, offset_t stride, offset_t stride_buf)
{
    offset_t k = 0;
    offset_t vk;
    z += stride_buf;
    for (offset_t q = 0; q < size; ++q) {
        scalar_t fq = mycast<scalar_t>(q);
        while ((k < size-1) && (*z < fq)) {
            z += stride_buf;
            ++k;
        }
        vk = v[k * stride_buf];
        f[q * stride] = d[vk * stride_buf]
                      + w2 * mycast<scalar_t>(square(q - vk));
    }
}

// Update the upper bound on the squared Euclidean distance.
//
// This function processes the data along a single dimension.
// Once it's been applied to all dimensions, `f` contains the squared
// L2 distance. Initially, `f` must contain "zero" in the background and
// "inf" in the foreground. The first pass can be performed using the L1
// sweep, which is faster.
//
// f          - [inp] Previous upper bound on the Euclidean distance
//              [out] Updated upper bound
// v          - [buf] Locations of parabolas in lower envelope
// z          - [buf] location of boundaries between parabolas
// d          - [buf] Working copy of `f`
// w2         - Squared voxel size along the current dimension
// size       - Number of voxels along the current dimension
// stride     - Stride of between two voxels along the current dimension (`f`)
// stride_buf - Stride of between two voxels along the current dimension (`d`)
template <typename offset_t, typename scalar_t>
__device__
void algo(scalar_t * f, offset_t * v, scalar_t * z, scalar_t * d, scalar_t w2,
          offset_t size, offset_t stride, offset_t stride_buf = 1)
{
    if (size == 1) return;

    for (offset_t q = 0; q < size; ++q, f += stride, d += stride_buf)
        *d = *f;
    f -= size * stride;
    d -= size * stride_buf;

    v[0] = 0;
    z[0] = -(1./0.);
    z[stride_buf] = 1./0.;
    scalar_t s;
    offset_t k = 0;
    scalar_t * zk;
    for (offset_t q=1; q < size; ++q) {
        zk = z + k * stride_buf;
        while (1) {
            s = intersection(d, v, w2, k, q, size, stride_buf);
            if ((k == 0) || (s > *zk))
                break;
            --k;
            zk -= stride_buf;
        }

        ++k;
        v[k * stride_buf] = q;
        z[k * stride_buf] = s;
        if (k < size-1)
            z[(k+1) * stride_buf] = 1./0.;
    }
    fillin(f, v, z, d, w2, size, stride, stride_buf);
}

} // namespace distance_e
} // namespace jf

#endif // JF_DISTANCE_E
