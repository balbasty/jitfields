/*
 * #include "distance_e.h"
 */
#include "cuda_switch.h"
#include "distance_euclidean.h"
#include "batch.h"

using namespace std;
using namespace jf;
using namespace jf::distance_e;

template <typename scalar_t, typename offset_t>
__global__ void kernel(scalar_t * f, char * buf, scalar_t w, int ndim,
                       const offset_t * size, const offset_t *  stride)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size, ndim-1);

    offset_t n = size[ndim-1];
    offset_t stride_buf = blockDim.x * gridDim.x;
    offset_t * v = reinterpret_cast<offset_t *>(buf);
    scalar_t * z = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * sizeof(offset_t));
    scalar_t * d = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * (sizeof(offset_t) + sizeof(scalar_t)));

    w = w*w;

    for (offset_t i=index; index < nthreads; index += stride_buf, i=index)
    {
        offset_t batch_offset = index2offset(i, ndim-1, size, stride);
        algo(f + batch_offset, v + i, z + i, d + i, w,
             n, stride[ndim-1], stride_buf);
    }
}
