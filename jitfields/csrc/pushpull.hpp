#ifndef JF_PUSHPULL_LOOP
#define JF_PUSHPULL_LOOP
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
void pull1d(scalar_t * out, scalar_t * inp, scalar_t * grid, int ndim,
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
void push1d(scalar_t * out, scalar_t * inp, scalar_t * grid, int ndim,
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
void count1d(scalar_t * out, scalar_t * grid, int ndim,
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
        if (!InFOV<extrapolate, one>::infov(x, nx)) {
            continue;
        }
        offset_t out_offset = index2offset(i, ndim-2, size_grid, stride_out);

        PushPull<one, IX, BX>::count(out + out_offset, x, nx, osx);
    }
}


template <spline::type IX, bound::type BX, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void grad1d(scalar_t * out, scalar_t * inp, scalar_t * grid, int ndim,
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
    scalar_t * inp, scalar_t * ginp,
    scalar_t * grid, int ndim,
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
    scalar_t * inp, scalar_t * ginp,
    scalar_t * grid, int ndim,
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
    scalar_t * gout, scalar_t * ginp,
    scalar_t * grid, int ndim,
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
    scalar_t * inp, scalar_t * ginp,
    scalar_t * grid, int ndim,
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

/***********************************************************************
 *
 *                                  2D
 *
 **********************************************************************/


template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void pull2d(scalar_t * out, scalar_t * inp, scalar_t * grid, int ndim,
            const offset_t * size_grid,
            const offset_t * size_splinc,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t isx = stride_inp[ndim-3];
    offset_t isy = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t out_offset = index2offset(i, ndim-1, size_grid, stride_out);
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
            out[out_offset] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-3, size_grid, stride_inp);

        PushPull<two, IX, BX, IY, BY>::pull(
            out + out_offset, inp + inp_offset,
            x, nx, isx, y, ny, isy, nc, osc, isc);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void push2d(scalar_t * out, scalar_t * inp, scalar_t * grid, int ndim,
            const offset_t * size_grid,
            const offset_t * size_splinc,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-3];
    offset_t osy = stride_out[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny))
            continue;
        offset_t inp_offset = index2offset(i, ndim-1, size_grid, stride_inp);
        offset_t out_offset = index2offset(i, ndim-3, size_grid, stride_out);

        PushPull<two, IX, BX, IY, BY>::push(
            out + out_offset, inp + inp_offset,
            x, nx, osx, y, ny, osy, nc, osc, isc);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void count2d(scalar_t * out, scalar_t * grid, int ndim,
             const offset_t * size_grid,
             const offset_t * size_splinc,
             const offset_t * stride_out,
             const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t osx = stride_out[ndim-3];
    offset_t osy = stride_out[ndim-2];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny))
            continue;
        offset_t out_offset = index2offset(i, ndim-3, size_grid, stride_out);

        PushPull<two, IX, BX, IY, BY>::count(
            out + out_offset, x, nx, osx, y, ny, osy);
    }
}


template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void grad2d(scalar_t * out, scalar_t * inp, scalar_t * grid, int ndim,
            const offset_t * size_grid,
            const offset_t * size_splinc,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t isx = stride_inp[ndim-3];
    offset_t isy = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t out_offset = index2offset(i, ndim-1, size_grid, stride_out);
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
            out[out_offset] = static_cast<scalar_t>(0);  // NaN?
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-3, size_grid, stride_inp);

        PushPull<two, IX, BX, IY, BY>::grad(
            out + out_offset, inp + inp_offset,
            x, nx, isx, y, ny, isy, nc, osc, isc);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void pull2d_backward(
    scalar_t * out, scalar_t * gout,
    scalar_t * inp, scalar_t * ginp,
    scalar_t * grid, int ndim,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_out,
    const offset_t * stride_gout,
    const offset_t * stride_inp,
    const offset_t * stride_ginp,
    const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-3];
    offset_t osy = stride_out[ndim-2];
    offset_t isx = stride_inp[ndim-3];
    offset_t isy = stride_inp[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t gsc = stride_ginp[ndim-1];
    offset_t grsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + grsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
            gout[gout_offset] = static_cast<scalar_t>(0);  // NaN?
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-3, size_grid, stride_inp);
        offset_t out_offset = index2offset(i, ndim-3, size_grid, stride_out);
        offset_t ginp_offset = index2offset(i, ndim-1, size_grid, stride_ginp);

        PushPull<two, IX, BX, IY, BY>::pull_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, osx, isx, y, ny, osy, isy, nc, osc, isc, gsc);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void push2d_backward(
    scalar_t * out, scalar_t * gout,
    scalar_t * inp, scalar_t * ginp,
    scalar_t * grid, int ndim,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_out,
    const offset_t * stride_gout,
    const offset_t * stride_inp,
    const offset_t * stride_ginp,
    const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-3];
    offset_t osy = stride_out[ndim-2];
    offset_t isx = stride_inp[ndim-3];
    offset_t isy = stride_inp[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t gsc = stride_ginp[ndim-1];
    offset_t grsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t out_offset = index2offset(i, ndim-3, size_grid, stride_out);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + grsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
            out[out_offset] = static_cast<scalar_t>(0);
            gout[gout_offset] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-3, size_grid, stride_inp);
        offset_t ginp_offset = index2offset(i, ndim-1, size_grid, stride_ginp);

        PushPull<two, IX, BX, IY, BY>::push_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, osx, isx, y, ny, osy, isy, nc, osc, isc, gsc);
    }
}


template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void count2d_backward(
    scalar_t * gout, scalar_t * ginp,
    scalar_t * grid, int ndim,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_gout,
    const offset_t * stride_ginp,
    const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t sx  = stride_ginp[ndim-3];
    offset_t sy  = stride_ginp[ndim-2];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
            gout[gout_offset] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t ginp_offset = index2offset(i, ndim-1, size_grid, stride_ginp);

        PushPull<two, IX, BX, IY, BY>::count_backward(
            gout + gout_offset, ginp + ginp_offset, x, nx, sx, y, ny, sy);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void grad2d_backward(
    scalar_t * out, scalar_t * gout,
    scalar_t * inp, scalar_t * ginp,
    scalar_t * grid, int ndim,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_out,
    const offset_t * stride_gout,
    const offset_t * stride_inp,
    const offset_t * stride_ginp,
    const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-3];
    offset_t osy = stride_out[ndim-2];
    offset_t isx = stride_inp[ndim-3];
    offset_t isy = stride_inp[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t gsc = stride_ginp[ndim-1];
    offset_t grsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + grsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
            gout[gout_offset] = static_cast<scalar_t>(0);  // NaN?
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-3, size_grid, stride_inp);
        offset_t out_offset = index2offset(i, ndim-3, size_grid, stride_out);
        offset_t ginp_offset = index2offset(i, ndim-1, size_grid, stride_ginp);

        PushPull<two, IX, BX, IY, BY>::grad_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, osx, isx, y, ny, osy, isy, nc, osc, isc, gsc);
    }
}

#if 0
template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          typename reduce_t, typename scalar_t, typename offset_t>
void pull2d(scalar_t * out, scalar_t * inp, scalar_t * grid, int ndim,
            const offset_t * size_out,
            const offset_t * size_inp,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_out, ndim-1);  // no outer loop across channels

    offset_t nx  = size_inp[ndim-3];
    offset_t ny  = size_inp[ndim-2];
    offset_t nc  = size_inp[ndim-1];
    offset_t isx = stride_inp[ndim-3];
    offset_t isy = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t inp_offset  = index2offset(i, ndim-3, size_out, stride_inp);
        offset_t out_offset  = index2offset(i, ndim-1, size_out, stride_out);
        offset_t grid_offset = index2offset(i, ndim-1, size_out, stride_grid);

        reduce_t x   = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y   = static_cast<reduce_t>(grid[grid_offset + gsc]);

        PushPull<two, IX, BX, IY, BY>::pull(
            out + out_offset, inp + inp_offset,
            x, y, nx, ny, isx, isy, nc, osc, isc);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void pull3d(scalar_t * out, scalar_t * inp, scalar_t * grid, int ndim,
            const offset_t * size_out,
            const offset_t * size_inp,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_out, ndim-1);  // no outer loop across channels

    offset_t nx  = size_inp[ndim-4];
    offset_t ny  = size_inp[ndim-3];
    offset_t nz  = size_inp[ndim-2];
    offset_t nc  = size_inp[ndim-1];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t inp_offset  = index2offset(i, ndim-4, size_out, stride_inp);
        offset_t out_offset  = index2offset(i, ndim-1, size_out, stride_out);
        offset_t grid_offset = index2offset(i, ndim-1, size_out, stride_grid);

        reduce_t x   = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y   = static_cast<reduce_t>(grid[grid_offset + gsc]);
        reduce_t z   = static_cast<reduce_t>(grid[grid_offset + gsc * 2]);

        PushPull<three, IX, BX, IY, BY, IZ, BZ>::pull(
            out + out_offset, inp + inp_offset,
            x, y, z, nx, ny, nz, isx, isy, isz, nc, osc, isc);
    }
}

template <int D, typename reduce_t, typename scalar_t, typename offset_t>
void pullnd(scalar_t * out, scalar_t * inp, scalar_t * grid, int ndim,
            const unsigned char * order,
            const unsigned char * bnd,
            const offset_t * size_out,
            const offset_t * size_inp,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_out, ndim-1);  // no outer loop across channels
    const spline::type * corder = reinterpret_cast<const spline::type *>(order);
    const bound::type  * cbnd   = reinterpret_cast<const bound::type *>(bnd);

    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];
    offset_t gsc = stride_grid[ndim-1];
    const offset_t * spatial_size = size_inp + ndim - D;
    const offset_t * spatial_strie = stride_inp + ndim - D;

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t inp_offset  = index2offset(i, ndim-1-D, size_out, stride_inp);
        offset_t out_offset  = index2offset(i, ndim-1, size_out, stride_out);
        offset_t grid_offset = index2offset(i, ndim-1, size_out, stride_grid);

        PushPull<D>::pull(
            out + out_offset, inp + inp_offset, grid + grid_offset,
            spatial_size, spatial_strie, nc, osc, isc, gsc,
            corder, cbnd);
    }
}
#endif

} // namespace pushpull
} // namespace jf

#endif // JF_PUSHPULL_LOOP
