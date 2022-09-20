#ifndef JF_DISTANCE_L1
#define JF_DISTANCE_L1
#include "cuda_switch.h"

namespace jf {
namespace distance_l1 {

template <typename offset_t, typename scalar_t>
__device__
void algo(scalar_t * f, offset_t size, offset_t stride, scalar_t w)
{
  if (size == 1) return;

  scalar_t tmp = *f;
  f += stride;
  for (offset_t i = 1; i < size; ++i, f += stride) {
     tmp = min(tmp + w, *f);
     *f = tmp;
  }
  f -= 2 * stride;
  for (offset_t i = size-2; i >= 0; --i, f -= stride) {
     tmp = min(tmp + w, *f);
     *f = tmp;
  }
}

} // namespace distance_l1
} // namespace jf

#endif // JF_DISTANCE_L1
