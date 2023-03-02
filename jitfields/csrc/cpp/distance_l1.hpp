#ifndef JF_DISTANCE_L1_LOOP
#define JF_DISTANCE_L1_LOOP
#include "../lib/cuda_switch.h"
#include "../lib/distance_l1.h"
#include "../lib/batch.h"

namespace jf {
namespace distance_l1 {

template <typename scalar_t, typename offset_t>
void loop(scalar_t * f, scalar_t w, int ndim,
          const offset_t * size, const offset_t *  stride)
{
    offset_t numel = prod(size, ndim-1);

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t batch_offset = index2offset(i, ndim-1, size, stride);
        algo(f + batch_offset, size[ndim-1], stride[ndim-1], w);
    }
}

template <typename scalar_t, typename offset_t>
void loop3d(scalar_t * f, scalar_t w,
            const offset_t * size, const offset_t *  stride)
{
    offset_t size_x = size[0], size_y = size[1], size_z = size[2];
    offset_t stride_x = stride[0], stride_y = stride[1], stride_z = stride[2];
    offset_t i, j;
    scalar_t *fi, *fj;
    for (i=0, fi=f; i < size_x; ++i, fi += stride_x)
    for (j=0, fj=fi; j < size_y; ++j, fj += stride_y)
        algo(fj, size_z, stride_z, w);
}

} // namespace distance _l1
} // namespace jf

#endif // JF_DISTANCE_L1_LOOP
