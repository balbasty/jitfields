/* TODO
 * - implement special case (order=1 + scale=2) for dim 2 and 3
 * - check if using an inner loop across batch elements is more efficient
 *   (we currently use an outer loop, so we recompute indices many times)
 */

#include "cuda_switch.h"
#include "spline.h"
#include "bounds.h"
#include "batch.h"
#include "restrict.h"

using namespace std;
using namespace jf;
using namespace jf::restrict;

template <spline::type IX, bound::type BX,
          typename scalar_t, typename offset_t>
__global__ void kernel1d(scalar_t * out, scalar_t * inp, int ndim,
                         scalar_t shift, const scalar_t * scale,
                         const offset_t * size_out,
                         const offset_t * size_inp,
                         const offset_t * stride_out,
                         const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x;
        offset_t batch_offset = index2offset_1d(i, ndim, size_out, stride_inp, x);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<one, zero, IX, BX>::restrict(
            out + out_offset, inp + batch_offset,
            x, size_inp[ndim-1], stride_inp[ndim-1],
            scale[ndim-1], shift);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          typename scalar_t, typename offset_t>
__global__ void kernel2d(scalar_t * out, scalar_t * inp, int ndim,
                         scalar_t shift, const scalar_t * scale,
                         const offset_t * size_out,
                         const offset_t * size_inp,
                         const offset_t * stride_out,
                         const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x, y;
        offset_t batch_offset = index2offset_2d(i, ndim, size_out, stride_inp, x, y);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<two, zero, IX, BX, IY, BY>::restrict(
            out + out_offset, inp + batch_offset,
            x, size_inp[ndim-2], stride_inp[ndim-2], scale[ndim-2],
            y, size_inp[ndim-1], stride_inp[ndim-1], scale[ndim-1],
            shift);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ,
          typename scalar_t, typename offset_t>
__global__ void kernel3d(scalar_t * out, scalar_t * inp, int ndim,
                         scalar_t shift, const scalar_t * scale,
                         const offset_t * size_out,
                         const offset_t * size_inp,
                         const offset_t * stride_out,
                         const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x, y, z;
        offset_t batch_offset = index2offset_3d(i, ndim, size_out, stride_inp, x, y, z);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<three, zero, IX, BX, IY, BY, IZ, BZ>::restrict(
            out + out_offset, inp + batch_offset,
            x, size_inp[ndim-3], stride_inp[ndim-3], scale[ndim-3],
            y, size_inp[ndim-2], stride_inp[ndim-2], scale[ndim-2],
            z, size_inp[ndim-1], stride_inp[ndim-1], scale[ndim-1],
            shift);
    }
}

template <int D, typename scalar_t, typename offset_t>
__global__ void kernelnd(scalar_t * out, scalar_t * inp, int ndim,
                         scalar_t shift, const scalar_t * scale,
                         const spline::type * order,
                         const bound::type * bnd,
                         const offset_t * size_out,
                         const offset_t * size_inp,
                         const offset_t * stride_out,
                         const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x[D];
        offset_t batch_offset = index2offset_nd(i, ndim, size_out, stride_inp, x, D);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<D>::restrict(
            out + out_offset, inp + batch_offset,
            x, size_inp + ndim - D, stride_inp + ndim - D,
            order, bnd, scale, shift);
    }
}

/* Special cases when scaling factor is bounded by (1, 2] */

template <spline::type IX, bound::type BX,
          typename scalar_t, typename offset_t>
__global__ void kernel21d(scalar_t * out, scalar_t * inp, int ndim,
                          scalar_t shift, const scalar_t * scale,
                          const offset_t * size_out,
                          const offset_t * size_inp,
                          const offset_t * stride_out,
                          const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x;
        offset_t batch_offset = index2offset_1d(i, ndim, size_out, stride_inp, x);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<one, two, IX, BX>::restrict(
            out + out_offset, inp + batch_offset,
            x, size_inp[ndim-1], stride_inp[ndim-1],
            scale[ndim-1], shift);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          typename scalar_t, typename offset_t>
__global__ void kernel22d(scalar_t * out, scalar_t * inp, int ndim,
                          scalar_t shift, const scalar_t * scale,
                          const offset_t * size_out,
                          const offset_t * size_inp,
                          const offset_t * stride_out,
                          const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x, y;
        offset_t batch_offset = index2offset_2d(i, ndim, size_out, stride_inp, x, y);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<two, two, IX, BX, IY, BY>::restrict(
            out + out_offset, inp + batch_offset,
            x, size_inp[ndim-2], stride_inp[ndim-2], scale[ndim-2],
            y, size_inp[ndim-1], stride_inp[ndim-1], scale[ndim-1],
            shift);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ,
          typename scalar_t, typename offset_t>
__global__ void kernel23d(scalar_t * out, scalar_t * inp, int ndim,
                          scalar_t shift, const scalar_t * scale,
                          const offset_t * size_out,
                          const offset_t * size_inp,
                          const offset_t * stride_out,
                          const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x, y, z;
        offset_t batch_offset = index2offset_3d(i, ndim, size_out, stride_inp, x, y, z);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<three, two, IX, BX, IY, BY, IZ, BZ>::restrict(
            out + out_offset, inp + batch_offset,
            x, size_inp[ndim-3], stride_inp[ndim-3], scale[ndim-3],
            y, size_inp[ndim-2], stride_inp[ndim-2], scale[ndim-2],
            z, size_inp[ndim-1], stride_inp[ndim-1], scale[ndim-1],
            shift);
    }
}
