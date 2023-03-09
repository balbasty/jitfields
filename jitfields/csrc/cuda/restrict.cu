/* TODO
 * - implement special case (order=1 + scale=2) for dim 2 and 3
 * - check if using an inner loop across batch elements is more efficient
 *   (we currently use an outer loop, so we recompute indices many times)
 */

#include "../lib/cuda_switch.h"
#include "../lib/spline.h"
#include "../lib/bounds.h"
#include "../lib/batch.h"
#include "../lib/restrict.h"

using namespace std;
using namespace jf;
using namespace jf::restrict;

template <int nbatch, int ndim,
          typename scalar_t, typename offset_t, typename reduce_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY,
          int U=zero>
__global__
void kernel(
    scalar_t * out,                 // (*batch, *shape) tensor
    const scalar_t * inp,           // (*batch, *shape) tensor
    reduce_t shift,
    const reduce_t * _scale,        // [*shape] vector
    const offset_t * _size_out,     // [*batch, *shape] vector
    const offset_t * _size_inp,     // [*batch, *shape] vector
    const offset_t * _stride_out,   // [*batch, *shape] vector
    const offset_t * _stride_inp)   // [*batch, *shape] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    offset_t scale      [ndim]; fillfrom<ndim>(scale,      _scale);
    offset_t size_out   [nall]; fillfrom<nall>(size_out,   _size_out);
    offset_t size_inp   [nall]; fillfrom<nall>(size_inp,   _size_inp);
    offset_t stride_out [nall]; fillfrom<nall>(stride_out, _stride_out);
    offset_t stride_inp [nall]; fillfrom<nall>(stride_inp, _stride_inp);

    offset_t numel = prod<nall>(size_out);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_nd<ndim,nall>(i, size_out, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size_out, stride_out);

        Multiscale<one, U, IX, BX, IY, BY, IZ, BZ>::restrict(
            out + out_offset, inp + inp_offset,
            loc, size_inp + nbatch, stride_inp + nbatch, scale, shift);
    }
}


template <int nbatch, int ndim,
          typename scalar_t, typename offset_t, typename reduce_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY,
          int U=zero>
__global__
void kernel2(
    scalar_t * out,                // (*batch, *shape) tensor
    const scalar_t * inp,          // (*batch, *shape) tensor
    reduce_t shift,
    const reduce_t * scale,        // [*shape] vector
    const offset_t * size_out,     // [*batch, *shape] vector
    const offset_t * size_inp,     // [*batch, *shape] vector
    const offset_t * stride_out,   // [*batch, *shape] vector
    const offset_t * stride_inp)   // [*batch, *shape] vector
{
    return kernel<nbatch, ndim, scalar_t, offset_t, reduce_t,
                  OX, BX, IY, BY, IZ, BZ, two>
        (out, inp, shift, scale, size_out, size_inp, stride_out, stride_inp);
}

template <int nbatch, int ndim,
          typename scalar_t, typename offset_t, typename reduce_t>
__global__
void kernelnd(
    scalar_t * out,                 // (*batch, *shape) tensor
    const scalar_t * inp,           // (*batch, *shape) tensor
    reduce_t shift,
    const reduce_t * _scale,        // [*shape] vector
    const unsigned char * _order,   // [*shape] vector
    const unsigned char * _bnd,     // [*shape] vector
    const offset_t * _size_out,     // [*batch, *shape] vector
    const offset_t * _size_inp,     // [*batch, *shape] vector
    const offset_t * _stride_out,   // [*batch, *shape] vector
    const offset_t * _stride_inp)   // [*batch, *shape] vector
{

    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int nall = ndim + nbatch;

    const spline::type * corder = reinterpret_cast<const spline::type *>(_order);
    const bound::type  * cbnd   = reinterpret_cast<const bound::type *>(_bnd);

    // copy vectors to the stack
    offset_t scale      [ndim]; fillfrom<ndim>(scale,      _scale);
    spline::type order  [ndim]; fillfrom<ndim>(order,      corder);
    bound::type  bnd    [ndim]; fillfrom<ndim>(bnd,        cbnd);
    offset_t size_out   [nall]; fillfrom<nall>(size_out,   _size_out);
    offset_t size_inp   [nall]; fillfrom<nall>(size_inp,   _size_inp);
    offset_t stride_out [nall]; fillfrom<nall>(stride_out, _stride_out);
    offset_t stride_inp [nall]; fillfrom<nall>(stride_inp, _stride_inp);

    offset_t numel = prod<nall>(size_out);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_nd<ndim,nall>(i, size_out, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size_out, stride_out);

        Multiscale<D>::restrict(
            out + out_offset, inp + inp_offset,
            x, size_inp + nbatch, stride_inp + nbatch,
            order, bnd, scale, shift);
    }
}
