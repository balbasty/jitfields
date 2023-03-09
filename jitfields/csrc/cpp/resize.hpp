#ifndef JF_RESIZE_LOOP
#define JF_RESIZE_LOOP
#include "../lib/cuda_switch.h"
#include "../lib/resize.h"
#include "../lib/batch.h"
#include "../lib/parallel.h"

namespace jf {
namespace resize {

template <int nbatch, int ndim,
          typename scalar_t, typename offset_t, typename reduce_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
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
    constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    reduce_t scale      [ndim]; fillfrom<ndim>(scale,      _scale);
    offset_t size_out   [nall]; fillfrom<nall>(size_out,   _size_out);
    offset_t size_inp   [nall]; fillfrom<nall>(size_inp,   _size_inp);
    offset_t stride_out [nall]; fillfrom<nall>(stride_out, _stride_out);
    offset_t stride_inp [nall]; fillfrom<nall>(stride_inp, _stride_inp);

    offset_t numel = prod<nall>(size_out);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_nd<ndim,nall>(i, size_out, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size_out, stride_out);

        Multiscale<ndim, IX, BX, IY, BY, IZ, BZ>::resize(
            out + out_offset, inp + inp_offset,
            loc, size_inp + nbatch, stride_inp + nbatch,
            scale, shift);
    }});
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

    offset_t numel = prod<nall>(size_out);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_nd<ndim,nall>(i, size_out, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size_out, stride_out);

        Multiscale<ndim>::resize(
            out + out_offset, inp + inp_offset,
            loc, size_inp + nbatch, stride_inp + nbatch,
            order, bnd, scale, shift);
    }});
}

} // namespace resize
} // namespace jf

#endif // JF_RESIZE_LOOP
