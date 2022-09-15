using namespace std;

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