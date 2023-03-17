#include "../lib/cuda_switch.h"
#include "../lib/distance.h"
#include "../lib/batch.h"

using namespace std;
using namespace jf;
using namespace jf::distance_l1;


template <int ndim, typename scalar_t, typename offset_t>
__global__
void kernel(
    scalar_t * f,
    scalar_t w,
    const offset_t * _size,
    const offset_t * _stride)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int nbatch = ndim - 1;

    offset_t size   [ndim]; fillfrom<ndim>(size,   _size);
    offset_t stride [ndim]; fillfrom<ndim>(stride, _stride);

    offset_t n = size[nbatch];
    offset_t s = stride[nbatch];

    offset_t numel = prod<nbatch>(size);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t offset = index2offset<nbatch>(i, size, stride);
        algo(f + offset, n, s, w);
    }
}
