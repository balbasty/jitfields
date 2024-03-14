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

    using bound_utils_x  = bound::utils<BX>;
    using bound_utils_y  = bound::utils<BY>;
    using bound_utils_z  = bound::utils<BZ>;
    constexpr int nall = ndim + nbatch;
    constexpr int spline_order_x = static_cast<int>(IX);
    constexpr int spline_order_y = static_cast<int>(IY);
    constexpr int spline_order_z = static_cast<int>(IZ);
    constexpr int padding_x = (spline_order_x + 1)/2;
    constexpr int padding_y = (spline_order_y + 1)/2;
    constexpr int padding_z = (spline_order_z + 1)/2;

    // copy vectors to the stack
    reduce_t scale      [ndim]; fillfrom<ndim>(scale,      _scale);
    offset_t size_out   [nall]; fillfrom<nall>(size_out,   _size_out);
    offset_t size_inp   [nall]; fillfrom<nall>(size_inp,   _size_inp);
    offset_t stride_out [nall]; fillfrom<nall>(stride_out, _stride_out);
    offset_t stride_inp [nall]; fillfrom<nall>(stride_inp, _stride_inp);

    offset_t fullsize[nall]; fillfrom<nall>(fullsize, size_out);
    if (ndim > 0) fullsize[nbatch]   += 2 * padding_x;
    if (ndim > 1) fullsize[nbatch+1] += 2 * padding_y;
    if (ndim > 2) fullsize[nbatch+2] += 2 * padding_z;

    offset_t numel = prod<nall>(fullsize);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t inp_offset = index2offset<nbatch>(i, size_out, stride_inp);

        signed char sgn = 1;
        offset_t loc[nall]; index2sub<nall>(i, fullsize, loc);
        offset_t sub[nall]; fillfrom<nall>(sub, loc);
        if (ndim > 0) {
            loc[nbatch]   -= padding_x;
            sgn           *= bound_utils_x::sign(loc[nbatch],  size_out[nbatch]);
            sub[nbatch]    = bound_utils_x::index(loc[nbatch], size_out[nbatch]);
        }
        if (ndim > 1) {
            loc[nbatch+1] -= padding_y;
            sgn           *= bound_utils_y::sign(loc[nbatch+1],  size_out[nbatch+1]);
            sub[nbatch+1]  = bound_utils_y::index(loc[nbatch+1], size_out[nbatch+1]);
        }
        if (ndim > 2) {
            loc[nbatch+2] -= padding_z;
            sgn           *= bound_utils_z::sign(loc[nbatch+2],  size_out[nbatch+2]);
            sub[nbatch+2]  = bound_utils_z::index(loc[nbatch+2], size_out[nbatch+2]);
        }
        if (!sgn) continue;

        offset_t out_offset = sub2offset<nall>(sub, stride_out);

        Multiscale<ndim, U, IX, IY, IZ>::restrict(
            out + out_offset, inp + inp_offset,
            loc + nbatch, size_inp + nbatch, stride_inp + nbatch,
            scale, shift, sgn);
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
                  IX, BX, IY, BY, IZ, BZ, two>
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
    reduce_t scale      [ndim]; fillfrom<ndim>(scale,      _scale);
    spline::type order  [ndim]; fillfrom<ndim>(order,      corder);
    bound::type  bnd    [ndim]; fillfrom<ndim>(bnd,        cbnd);
    offset_t size_out   [nall]; fillfrom<nall>(size_out,   _size_out);
    offset_t size_inp   [nall]; fillfrom<nall>(size_inp,   _size_inp);
    offset_t stride_out [nall]; fillfrom<nall>(stride_out, _stride_out);
    offset_t stride_inp [nall]; fillfrom<nall>(stride_inp, _stride_inp);

    offset_t fullsize[nall]; fillfrom<nall>(fullsize, size_out);
    for (int d=0; d < ndim; ++d)
        fullsize[nbatch+d] += 2 * ((static_cast<int>(order[d]) + 1) / 2);

    offset_t numel = prod<nall>(fullsize);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t inp_offset = index2offset<nbatch>(i, size_out, stride_inp);

        signed char sgn = 1;
        offset_t loc[nall]; index2sub<nall>(i, fullsize, loc);
        offset_t sub[nall]; fillfrom<nall>(sub, loc);
        for (int d=0; d < ndim; ++d) {
            loc[nbatch+d]   -= (static_cast<int>(order[d]) + 1) / 2;
            sgn             *= bound::sign(bnd[d], loc[nbatch+d],  size_out[nbatch+d]);
            sub[nbatch+d]    = bound::index(bnd[d], loc[nbatch+d], size_out[nbatch+d]);
        }
        if (!sgn) continue;

        offset_t out_offset = sub2offset<nall>(sub, stride_out);

        Multiscale<ndim>::restrict(
            out + out_offset, inp + inp_offset,
            loc + nbatch, size_inp + nbatch, stride_inp + nbatch,
            order, scale, shift, sgn);
    }
}
