#ifndef JF_PUSHPULL_LOOP2D
#define JF_PUSHPULL_LOOP2D
#include "cuda_switch.h"
#include "pushpull.h"
#include "batch.h"

namespace jf {
namespace pushpull {

/***********************************************************************
 *
 *                                  2D
 *
 **********************************************************************/


template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void pull2d(scalar_t * out,
            const scalar_t * inp, const scalar_t * grid, int ndim,
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
void push2d(scalar_t * out,
            const scalar_t * inp, const scalar_t * grid, int ndim,
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
void count2d(scalar_t * out, const scalar_t * grid, int ndim,
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
void grad2d(scalar_t * out,
            const scalar_t * inp, const scalar_t * grid, int ndim,
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
    offset_t osg = stride_out[ndim];
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
            x, nx, isx, y, ny, isy, nc, osc, isc, osg);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void pull2d_backward(
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

    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-3];
    offset_t osy = stride_out[ndim-2];
    offset_t isx = stride_inp[ndim-3];
    offset_t isy = stride_inp[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t isg = stride_ginp[ndim-1];
    offset_t osg = stride_gout[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
            gout[gout_offset]       = static_cast<scalar_t>(0);
            gout[gout_offset + osg] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-3, size_grid, stride_inp);
        offset_t out_offset = index2offset(i, ndim-3, size_grid, stride_out);
        offset_t ginp_offset = index2offset(i, ndim-1, size_grid, stride_ginp);

        PushPull<two, IX, BX, IY, BY>::pull_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, osx, isx, y, ny, osy, isy, nc, osc, isc, osg, isg);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void push2d_backward(
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

    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t isx = stride_inp[ndim-3];
    offset_t isy = stride_inp[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t isg = stride_ginp[ndim-1];
    offset_t osg = stride_gout[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t out_offset  = index2offset(i, ndim-1, size_grid, stride_out);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
            out[out_offset]         = static_cast<scalar_t>(0);
            gout[gout_offset]       = static_cast<scalar_t>(0);
            gout[gout_offset + osg] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-1, size_grid, stride_inp);
        offset_t ginp_offset = index2offset(i, ndim-3, size_grid, stride_ginp);

        PushPull<two, IX, BX, IY, BY>::push_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, isx, y, ny, isy, nc, osc, isc, osg, isg);
    }
}


template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void count2d_backward(
    scalar_t * gout, const scalar_t * ginp,
    const scalar_t * grid, int ndim,
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
    offset_t osg = stride_gout[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
            gout[gout_offset]       = static_cast<scalar_t>(0);
            gout[gout_offset + osg] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t ginp_offset = index2offset(i, ndim-3, size_grid, stride_ginp);

        PushPull<two, IX, BX, IY, BY>::count_backward(
            gout + gout_offset, ginp + ginp_offset, x, nx, sx, y, ny, sy, osg);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void grad2d_backward(
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

    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-3];
    offset_t osy = stride_out[ndim-2];
    offset_t isx = stride_inp[ndim-3];
    offset_t isy = stride_inp[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t osg = stride_gout[ndim-1];
    offset_t isg = stride_ginp[ndim];
    offset_t gsc = stride_ginp[ndim-1];
    offset_t grsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + grsc]);
        if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
            gout[gout_offset]       = static_cast<scalar_t>(0);
            gout[gout_offset + osg] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-3, size_grid, stride_inp);
        offset_t out_offset = index2offset(i, ndim-3, size_grid, stride_out);
        offset_t ginp_offset = index2offset(i, ndim-1, size_grid, stride_ginp);

        PushPull<two, IX, BX, IY, BY>::grad_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, osx, isx, y, ny, osy, isy, nc, osc, isc, gsc, osg, isg);
    }
}

} // namespace pushpull
} // namespace jf

#endif // JF_PUSHPULL_LOOP2D
