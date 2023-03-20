#ifndef JF_DISTANCE_E_LOOP
#define JF_DISTANCE_E_LOOP
#include "../lib/cuda_switch.h"
#include "../lib/distance.h"
#include "../lib/batch.h"
#include "../lib/parallel.h"

namespace jf {
namespace distance_e {

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
    w = w*w;

    offset_t numel = prod<nbatch>(size);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    auto v = new offset_t[n];
    auto z = new scalar_t[n];
    auto d = new scalar_t[n];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset = index2offset<nbatch>(i, size, stride);
        algo(f + offset, v, z, d, w, n, s);
    }
    delete[] v;
    delete[] z;
    delete[] d;
    });
}

} // namespace distance_e
} // namespace jf

#endif // _LOOP
