#ifndef JF_RESTRICT_LOOP
#define JF_RESTRICT_LOOP
#include "../lib/cuda_switch.h"
#include "../lib/restrict.h"
#include "../lib/batch.h"
#include "../lib/parallel.h"

namespace jf {
namespace restrict {

template <int nbatch, int ndim,
          typename scalar_t, typename offset_t, typename reduce_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY,
          int U=zero>
void loop(
    scalar_t * out,                 // (*batch, *shape) tensor
    const scalar_t * inp,           // (*batch, *shape) tensor
    reduce_t shift,
    const reduce_t * _scale,        // [*shape] vector
    const offset_t * _size_out,     // [*batch, *shape] vector
    const offset_t * _size_inp,     // [*batch, *shape] vector
    const offset_t * _stride_out,   // [*batch, *shape] vector
    const offset_t * _stride_inp)   // [*batch, *shape] vector
{
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

    if ( jf::has_atomic_add<scalar_t>::value )
    {
        offset_t numel = prod<nall>(size_out);
        parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
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

            Multiscale<ndim, U, IX, BX, IY, BY, IZ, BZ>::restrict(
                out + out_offset, inp + inp_offset,
                loc + nbatch, size_inp + nbatch, stride_inp + nbatch,
                scale, shift, sgn);
        }});
    }
    else
    {
        offset_t numel_batch   = prod<nbatch>(fullsize);
        offset_t numel_spatial = prod<ndim>(fullsize+nbatch);
        long grain_size = max(GRAIN_SIZE/numel_spatial, 1L);
        parallel_for(0, numel_batch, grain_size, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t inp_offset = index2offset<nbatch>(i, size_out, stride_inp);
            offset_t out_offset0 = index2offset<nbatch>(i, size_out, stride_out);
            for (offset_t j=0; j < numel_spatial; ++j)
            {
                signed char sgn = 1;
                offset_t loc[ndim]; index2sub<ndim>(j, fullsize + nbatch, loc);
                offset_t sub[ndim];
                if (ndim > 0) {
                    loc[0]   -= padding_x;
                    sgn      *= bound_utils_x::sign(loc[0],  size_out[nbatch]);
                    sub[0]    = bound_utils_x::index(loc[0], size_out[nbatch]);
                }
                if (ndim > 1) {
                    loc[1] -= padding_y;
                    sgn    *= bound_utils_y::sign(loc[1],  size_out[nbatch+1]);
                    sub[1]  = bound_utils_y::index(loc[1], size_out[nbatch+1]);
                }
                if (ndim > 2) {
                    loc[2] -= padding_z;
                    sgn    *= bound_utils_z::sign(loc[2],  size_out[nbatch+2]);
                    sub[2]  = bound_utils_z::index(loc[2], size_out[nbatch+2]);
                }
                if (!sgn) continue;

                offset_t out_offset = out_offset0
                    + sub2offset<ndim>(sub, stride_out + nbatch);

                Multiscale<ndim, U, IX, BX, IY, BY, IZ, BZ>::restrict(
                    out + out_offset, inp + inp_offset,
                    loc, size_inp + nbatch, stride_inp + nbatch,
                    scale, shift, sgn);
            }
        }});
    }
}

// Special cases when scaling factor is bounded by (1, 2]
template <int nbatch, int ndim,
          typename scalar_t, typename offset_t, typename reduce_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
void loop2(
    scalar_t * out,                 // (*batch, *shape) tensor
    const scalar_t * inp,           // (*batch, *shape) tensor
    reduce_t shift,
    const reduce_t * scale,        // [*shape] vector
    const offset_t * size_out,     // [*batch, *shape] vector
    const offset_t * size_inp,     // [*batch, *shape] vector
    const offset_t * stride_out,   // [*batch, *shape] vector
    const offset_t * stride_inp)   // [*batch, *shape] vector
{
    return loop<nbatch, ndim, scalar_t, offset_t, reduce_t,
                IX, BX, IY, BY, IZ, BZ, two>
        (out, inp, shift, scale, size_out, size_inp, stride_out, stride_inp);
}

template <int nbatch, int ndim,
          typename scalar_t, typename offset_t, typename reduce_t>
void loopnd(
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

    if ( jf::has_atomic_add<scalar_t>::value )
    {
        offset_t numel = prod<nall>(size_out);
        parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
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
                order, bnd, scale, shift, sgn);
        }});
    }
    else
    {
        offset_t numel_batch   = prod<nbatch>(fullsize);
        offset_t numel_spatial = prod<ndim>(fullsize+nbatch);
        long grain_size = max(GRAIN_SIZE/numel_spatial, 1L);
        parallel_for(0, numel_batch, grain_size, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t inp_offset = index2offset<nbatch>(i, size_out, stride_inp);
            offset_t out_offset0 = index2offset<nbatch>(i, size_out, stride_out);
            for (offset_t j=0; j < numel_spatial; ++j)
            {
                signed char sgn = 1;
                offset_t loc[ndim]; index2sub<ndim>(j, fullsize + nbatch, loc);
                offset_t sub[ndim];
                for (int d=0; d < ndim; ++d) {
                    loc[d]   -= (static_cast<int>(order[d]) + 1) / 2;
                    sgn      *= bound::sign(bnd[d], loc[d],  size_out[nbatch+d]);
                    sub[d]    = bound::index(bnd[d], loc[d], size_out[nbatch+d]);
                }
                if (!sgn) continue;

                offset_t out_offset = out_offset0
                    + sub2offset<ndim>(sub, stride_out + nbatch);

                Multiscale<ndim>::restrict(
                    out + out_offset, inp + inp_offset,
                    loc, size_inp + nbatch, stride_inp + nbatch,
                    order, bnd, scale, shift, sgn);
            }
        }});
    }
}

} // namespace restrict
} // namespace jf

#endif // JF_RESTRICT_LOOP
