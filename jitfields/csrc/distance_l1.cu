/*
 * #include "distance_l1.h"
 */
#include "cuda_switch.h"
#include "distance_l1.h"
#include "batch.h"

using namespace std;
using namespace jf;
using namespace jf::distance_l1;


template <typename scalar_t, typename offset_t>
__global__ void kernel(scalar_t * f, scalar_t w, int ndim,
                       const offset_t * size, const offset_t *  stride)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size, ndim-1);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t batch_offset = index2offset(i, ndim-1, size, stride);
        algo(f + batch_offset, size[ndim-1], stride[ndim-1], w);
    }
}
