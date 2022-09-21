#ifndef JF_SPLINC_LOOP
#define JF_SPLINC_LOOP
#include "cuda_switch.h"
#include "splinc.h"
#include "batch.h"

namespace jf {
namespace splinc {

template <bound::type B, typename scalar_t, typename offset_t>
void loop(scalar_t * inp, int ndim,
          const offset_t * size, const offset_t * stride,
          const double * poles, int npoles)
{
    offset_t numel = prod(size, ndim-1);

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t offset = index2offset(i, ndim-1, size, stride);
        splinc::filter<B>(inp + offset, size[ndim-1], stride[ndim-1],
                          poles, npoles);
    }
}

} // namespace splinc
} // namespace jf

#endif // JF_SPLINC_LOOP
