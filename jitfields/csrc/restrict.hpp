#ifndef JF_RESTRICT_LOOP
#define JF_RESTRICT_LOOP
#include "cuda_switch.h"
#include "restrict.h"
#include "batch.h"

namespace jf {
namespace restrict {

template <spline::type IX, bound::type BX,
          typename scalar_t, typename offset_t>
void loop1d(scalar_t * out, const scalar_t * inp, int ndim,
            scalar_t shift, const scalar_t * scale,
            const offset_t * size_out,
            const offset_t * size_inp,
            const offset_t * stride_out,
            const offset_t * stride_inp)
{
    offset_t numel = prod(size_out, ndim);

    for (offset_t i=0; i < numel; ++i)
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
void loop2d(scalar_t * out, const scalar_t * inp, int ndim,
            scalar_t shift, const scalar_t * scale,
            const offset_t * size_out,
            const offset_t * size_inp,
            const offset_t * stride_out,
            const offset_t * stride_inp)
{
    offset_t numel = prod(size_out, ndim);

    for (offset_t i=0; i < numel; ++i)
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
void loop3d(scalar_t * out, const scalar_t * inp, int ndim,
            scalar_t shift, const scalar_t * scale,
            const offset_t * size_out,
            const offset_t * size_inp,
            const offset_t * stride_out,
            const offset_t * stride_inp)
{
    offset_t numel = prod(size_out, ndim);

    for (offset_t i=0; i < numel; ++i)
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
void loopnd(scalar_t * out, const scalar_t * inp, int ndim,
            scalar_t shift, const scalar_t * scale,
            const unsigned char * order,
            const unsigned char * bnd,
            const offset_t * size_out,
            const offset_t * size_inp,
            const offset_t * stride_out,
            const offset_t * stride_inp)
{
    offset_t numel = prod(size_out, ndim);
    const spline::type * corder = reinterpret_cast<const spline::type *>(order);
    const bound::type  * cbnd   = reinterpret_cast<const bound::type *>(bnd);

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t x[D];
        offset_t batch_offset = index2offset_nd(i, ndim, size_out, stride_inp, x, D);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<D>::restrict(
            out + out_offset, inp + batch_offset,
            x, size_inp + ndim - D, stride_inp + ndim - D,
            corder, cbnd, scale, shift);
    }
}

/* Special cases when scaling factor is bounded by (1, 2] */

template <spline::type IX, bound::type BX,
          typename scalar_t, typename offset_t>
void loop21d(scalar_t * out, const scalar_t * inp, int ndim,
             scalar_t shift, const scalar_t * scale,
             const offset_t * size_out,
             const offset_t * size_inp,
             const offset_t * stride_out,
             const offset_t * stride_inp)
{
    offset_t numel = prod(size_out, ndim);

    for (offset_t i=0; i < numel; ++i)
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
void loop22d(scalar_t * out, const scalar_t * inp, int ndim,
             scalar_t shift, const scalar_t * scale,
             const offset_t * size_out,
             const offset_t * size_inp,
             const offset_t * stride_out,
             const offset_t * stride_inp)
{
    offset_t numel = prod(size_out, ndim);

    for (offset_t i=0; i < numel; ++i)
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
void loop23d(scalar_t * out, const scalar_t * inp, int ndim,
             scalar_t shift, const scalar_t * scale,
             const offset_t * size_out,
             const offset_t * size_inp,
             const offset_t * stride_out,
             const offset_t * stride_inp)
{
    offset_t numel = prod(size_out, ndim);

    for (offset_t i=0; i < numel; ++i)
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


} // namespace restrict
} // namespace jf

#endif // JF_RESTRICT_LOOP
