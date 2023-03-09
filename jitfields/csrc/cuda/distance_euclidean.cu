#include "../lib/cuda_switch.h"
#include "../lib/distance.h"
#include "../lib/batch.h"

using namespace std;
using namespace jf;
using namespace jf::distance_e;

template <int ndim, typename scalar_t, typename offset_t>
__global__ void kernel(
    scalar_t * f,
    char * buf,
    scalar_t w,
    const offset_t * _size,
    const offset_t * _stride)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int nbatch = ndim - 1;

    offset_t size   [ndim]; fillfrom<ndim>(size,   _size);
    offset_t stride [ndim]; fillfrom<ndim>(stride, _stride);

    offset_t n = size[nbatch];
    offset_t s = size[nbatch];
    w *= w;

    offset_t stride_buf = blockDim.x * gridDim.x;
    offset_t * v = reinterpret_cast<offset_t *>(buf);
    scalar_t * z = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * sizeof(offset_t));
    scalar_t * d = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * (sizeof(offset_t) + sizeof(scalar_t)));
    v += index;
    z += index;
    d += index;

    offset_t numel = prod<nbatch>(size);
    for (offset_t i=index; index < numel; index += stride_buf, i=index)
    {
        offset_t offset = index2offset<nbatch>(i, size, stride);
        algo(f + offset, v, z, d, w, n, s, stride_buf);
    }
}
