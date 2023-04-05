#ifndef JF_DISTANCE_L1_LOOP
#define JF_DISTANCE_L1_LOOP
#include "../lib/cuda_switch.h"
#include "../lib/distance.h"
#include "../lib/batch.h"
#include "../lib/parallel.h"

namespace jf {
namespace distance_l1 {

template <int ndim, typename scalar_t, typename offset_t>
void loop(
    scalar_t * f,
    scalar_t w,
    const offset_t * _size,
    const offset_t * _stride)
{
    constexpr int nbatch = ndim - 1;
    offset_t size   [ndim]; fillfrom<ndim>(size,   _size);
    offset_t stride [ndim]; fillfrom<ndim>(stride, _stride);
    offset_t n = size[nbatch];
    offset_t s = stride[nbatch];

    offset_t numel = prod<nbatch>(size);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset = index2offset<nbatch>(i, size, stride);
        algo(f + offset, n, s, w);
    }});
}

} // namespace distance _l1
} // namespace jf

#endif // JF_DISTANCE_L1_LOOP
