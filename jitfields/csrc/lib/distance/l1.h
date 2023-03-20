// "Distance Transforms of Sampled Functions"
// Pedro F. Felzenszwalb & Daniel P. Huttenlocher
// Theory of Computing (2012)
// https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
#ifndef JF_DISTANCE_L1
#define JF_DISTANCE_L1
#include "../cuda_switch.h"
#include "../utils.h"

namespace jf {
namespace distance_l1 {

// Update the upper bound on the L1 distance.
//
// This function processes the data along a single dimension.
// Once it's been applied to all dimensions, `f` contains the L1 distance.
// Initially, `f` must contain "zero" in the background and "inf" in
// the foreground.
//
// f      - [inp] upper bound of the distance to nearest "0"
//        - [out] updated upper bound
// size   - Number of voxels along the current dimension
// stride - Stride between two voxels along the current dimension
// w      - Voxel size along the current dimension
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
