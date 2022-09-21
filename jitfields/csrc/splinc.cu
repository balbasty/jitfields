#include "cuda_switch.h"
#include "splinc.h"
#include "batch.h"

template <bound::type B, typename scalar_t, typename offset_t>
__global__ void kernel(scalar_t * inp, int ndim,
                       const offset_t * size, const offset_t * stride,
                       const double * poles, int npoles)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size, ndim-1);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t offset = index2offset(i, ndim-1, size, stride);
        splinc::filter<B>(inp + offset, size[ndim-1], stride[ndim-1],
                          poles, npoles);
    }
}
