#include "../lib/cuda_switch.h"
#include "../lib/splinc.h"
#include "../lib/bounds.h"
#include "../lib/batch.h"

using namespace jf;

template <int ndim, int npoles, bound::type B,
          typename scalar_t, typename offset_t, typename reduce_t>
__global__
void kernel(
    scalar_t * inp,
    const offset_t * size,
    const offset_t * stride,
    const reduce_t * poles)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;

    constexpr int nbatch = ndim - 1;
    reduce_t poles  [npoles];  fill<npoles>(poles, _poles);
    offset_t size   [ndim];    fill<ndim>(size,    _size);
    offset_t stride [ndim];    fill<ndim>(stride, _stride);

    offset_t numel = prod<nbatch>(size);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t offset = index2offset<nbatch>(i, size, stride);
        splinc::filter<B,npoles>(
            inp + offset, size[nbatch], stride[nbatch], poles, npoles);
    }
}
