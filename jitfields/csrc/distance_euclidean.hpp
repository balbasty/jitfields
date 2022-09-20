#ifndef JF_DISTANCE_E_LOOP
#define JF_DISTANCE_E_LOOP
#include "cuda_switch.h"
#include "distance_euclidean.h"

namespace jf {
namespace distance_e {

template <typename scalar_t, typename offset_t>
void loop(scalar_t * f, unsigned char * buf, scalar_t w, int ndim,
          const offset_t * size, const offset_t *  stride)
{
    offset_t numel = prod(size, ndim-1);

    offset_t n = size[ndim-1];

    offset_t stride_buf = 1;
    offset_t * v = reinterpret_cast<offset_t *>(buf);
    scalar_t * z = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * sizeof(offset_t));
    scalar_t * d = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * (sizeof(offset_t) + sizeof(scalar_t)));

    w = w*w;

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t batch_offset = index2offset(i, ndim-1, size, stride);
        algo(f + batch_offset, v + index, z + index, d + index, w,
             n, stride[ndim-1], stride_buf);
    }
}

template <typename scalar_t, typename offset_t>
void loop3d(scalar_t * f, unsigned char * buf, scalar_t w,
            const offset_t * size, const offset_t *  stride)
{
    offset_t size_x = size[0], size_y = size[1], size_z = size[2];
    offset_t stride_x = stride[0], stride_y = stride[1], stride_z = stride[2];

    offset_t * v = reinterpret_cast<offset_t *>(buf);
    scalar_t * z = reinterpret_cast<scalar_t *>(buf
                 + size_z * sizeof(offset_t));
    scalar_t * d = reinterpret_cast<scalar_t *>(buf
                 + size_z * (sizeof(offset_t) + sizeof(scalar_t)));

    offset_t i, j;
    scalar_t *fi, *fj;
    for (i=0, fi=f; i < size_x; ++i, fi += stride_x)
    for (j=0, fj=fi; j < size_y; ++j, fj += stride_y)
        algo(fj, v, z, d, w, size_z, stride_z, static_cast<offset_t>(1));
}

} // namespace distance_e
} // namespace jf

#endif // _LOOP
