#ifndef JF_PUSHPULL_LOOP1D
#define JF_PUSHPULL_LOOP1D
#include "cuda_switch.h"
#include "pushpull.h"
#include "batch.h"

namespace jf {
namespace pushpull {

/***********************************************************************
 *
 *                                  1D
 *
 **********************************************************************/

template <spline::type IX, bound::type BX, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void pull1d(scalar_t * out,
            const scalar_t * inp, const scalar_t * grid, int ndim,
            const offset_t * size_grid,
            const offset_t * size_splinc,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t isx = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t out_offset = index2offset(i, ndim-1, size_grid, stride_out);
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        if (!InFOV<extrapolate, one>::infov(x, nx)) {
            out[out_offset] = static_cast<scalar_t>(0);  // NaN?
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-2, size_grid, stride_inp);

        PushPull<one, IX, BX>::pull(out + out_offset, inp + inp_offset,
                                    x, nx, isx, nc, osc, isc);
    }
}

template <spline::type IX, bound::type BX, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void push1d(scalar_t * out,
            const scalar_t * inp, const scalar_t * grid, int ndim,
            const offset_t * size_grid,
            const offset_t * size_splinc,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        if (!InFOV<extrapolate, one>::infov(x, nx))
            continue;
        offset_t inp_offset = index2offset(i, ndim-1, size_grid, stride_inp);
        offset_t out_offset = index2offset(i, ndim-2, size_grid, stride_out);

        PushPull<one, IX, BX>::push(out + out_offset, inp + inp_offset,
                                    x, nx, osx, nc, osc, isc);
    }
}

template <spline::type IX, bound::type BX, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void count1d(scalar_t * out,
             const scalar_t * grid, int ndim,
             const offset_t * size_grid,
             const offset_t * size_splinc,
             const offset_t * stride_out,
             const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-2];
    offset_t osx = stride_out[ndim-2];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        if (!InFOV<extrapolate, one>::infov(x, nx))
            continue;
        offset_t out_offset = index2offset(i, ndim-2, size_grid, stride_out);

        PushPull<one, IX, BX>::count(out + out_offset, x, nx, osx);
    }
}


template <spline::type IX, bound::type BX, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void grad1d(scalar_t * out,
            const scalar_t * inp, const scalar_t * grid, int ndim,
            const offset_t * size_grid,
            const offset_t * size_splinc,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t isx = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t out_offset  = index2offset(i, ndim-1, size_grid, stride_out);
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        if (!InFOV<extrapolate, one>::infov(x, nx)) {
            out[out_offset] = static_cast<scalar_t>(0);  // NaN?
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-2, size_grid, stride_inp);

        PushPull<one, IX, BX>::grad(out + out_offset, inp + inp_offset,
                                    x, nx, isx, nc, osc, isc);
    }
}

template <spline::type IX, bound::type BX, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void pull1d_backward(
    scalar_t * out, scalar_t * gout,
    const scalar_t * inp, const scalar_t * ginp,
    const scalar_t * grid, int ndim,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_out,
    const offset_t * stride_gout,
    const offset_t * stride_inp,
    const offset_t * stride_ginp,
    const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-2];
    offset_t isx = stride_inp[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t gsc = stride_ginp[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        if (!InFOV<extrapolate, one>::infov(x, nx)) {
            gout[gout_offset] = static_cast<scalar_t>(0);  // NaN?
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-2, size_grid, stride_inp);
        offset_t out_offset = index2offset(i, ndim-2, size_grid, stride_out);
        offset_t ginp_offset = index2offset(i, ndim-1, size_grid, stride_ginp);

        PushPull<one, IX, BX>::pull_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, osx, isx, nc, osc, isc, gsc);
    }
}

template <spline::type IX, bound::type BX, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void push1d_backward(
    scalar_t * out, scalar_t * gout,
    const scalar_t * inp, const scalar_t * ginp,
    const scalar_t * grid, int ndim,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_out,
    const offset_t * stride_gout,
    const offset_t * stride_inp,
    const offset_t * stride_ginp,
    const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t sx  = stride_ginp[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t gsc = stride_ginp[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t out_offset  = index2offset(i, ndim-1, size_grid, stride_out);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        if (!InFOV<extrapolate, one>::infov(x, nx)) {
            out[out_offset]   = static_cast<scalar_t>(0);
            gout[gout_offset] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset  = index2offset(i, ndim-1, size_grid, stride_inp);
        offset_t ginp_offset = index2offset(i, ndim-2, size_grid, stride_ginp);

        PushPull<one, IX, BX>::push_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, sx, nc, osc, isc, gsc);
    }
}


template <spline::type IX, bound::type BX, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void count1d_backward(
    scalar_t * gout, const scalar_t * ginp,
    const scalar_t * grid, int ndim,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_gout,
    const offset_t * stride_ginp,
    const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-2];
    offset_t sx  = stride_ginp[ndim-2];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        if (!InFOV<extrapolate, one>::infov(x, nx)) {
            gout[gout_offset] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t ginp_offset = index2offset(i, ndim-2, size_grid, stride_ginp);

        PushPull<one, IX, BX>::count_backward(
            gout + gout_offset, ginp + ginp_offset, x, nx, sx);
    }
}

template <spline::type IX, bound::type BX, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void grad1d_backward(
    scalar_t * out, scalar_t * gout,
    const scalar_t * inp, const scalar_t * ginp,
    const scalar_t * grid, int ndim,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_out,
    const offset_t * stride_gout,
    const offset_t * stride_inp,
    const offset_t * stride_ginp,
    const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-2];
    offset_t isx = stride_inp[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t gsc = stride_ginp[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        if (!InFOV<extrapolate, one>::infov(x, nx)) {
            gout[gout_offset] = static_cast<scalar_t>(0);  // NaN?
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-2, size_grid, stride_inp);
        offset_t out_offset = index2offset(i, ndim-2, size_grid, stride_out);
        offset_t ginp_offset = index2offset(i, ndim-1, size_grid, stride_ginp);

        PushPull<one, IX, BX>::grad_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, osx, isx, nc, osc, isc, gsc);
    }
}

} // namespace pushpull
} // namespace jf

#endif // JF_PUSHPULL_LOOP1D
