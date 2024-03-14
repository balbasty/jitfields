#ifndef JF_SPLINC_LOOP
#define JF_SPLINC_LOOP
#include "../lib/cuda_switch.h"
#include "../lib/splinc.h"
#include "../lib/batch.h"
#include "../lib/utils.h"
#include "../lib/parallel.h"

namespace jf {
namespace splinc {

template <int nbatch, int npoles, bound::type B,
          typename scalar_t, typename offset_t, typename reduce_t>
void loop(
    scalar_t * inp,
    const offset_t * _size,
    const offset_t * _stride,
    const reduce_t * _poles)
{
    constexpr int ndim = nbatch + 1;
    reduce_t poles  [npoles];  fillfrom<npoles>(poles, _poles);
    offset_t size   [ndim];    fillfrom<ndim>(size,    _size);
    offset_t stride [ndim];    fillfrom<ndim>(stride, _stride);

    offset_t numel = prod<nbatch>(size);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset = index2offset<nbatch>(i, size, stride);
        splinc::filter<B,npoles>(
            inp + offset, size[nbatch], stride[nbatch], poles);
    }});
}

} // namespace splinc
} // namespace jf

#endif // JF_SPLINC_LOOP
