#ifndef JF_PUSHPULL_LOOP3D
#define JF_PUSHPULL_LOOP3D
#include "cuda_switch.h"
#include "pushpull.h"
#include "batch.h"

namespace jf {
namespace pushpull {

/***********************************************************************
 *
 *                                  3D
 *
 **********************************************************************/


template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void pull3d(scalar_t * out,
            const scalar_t * inp, const scalar_t * grid, int ndim,
            const offset_t * size_grid,
            const offset_t * size_splinc,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-4];
    offset_t ny  = size_splinc[ndim-3];
    offset_t nz  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t out_offset = index2offset(i, ndim-1, size_grid, stride_out);
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        reduce_t z = static_cast<reduce_t>(grid[grid_offset + gsc * 2]);
        if (!InFOV<extrapolate, three>::infov(x, y, z, nx, ny, nz)) {
            for (offset_t c=0; c<nc; ++c)
                out[out_offset + c * osc]   = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-4, size_grid, stride_inp);

        PushPull<three, IX, BX, IY, BY, IZ, BZ>::pull(
            out + out_offset, inp + inp_offset,
            x, nx, isx, y, ny, isy, z, nz, isz, nc, osc, isc);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void push3d(scalar_t * out,
            const scalar_t * inp, const scalar_t * grid, int ndim,
            const offset_t * size_grid,
            const offset_t * size_splinc,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-4];
    offset_t ny  = size_splinc[ndim-3];
    offset_t nz  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-4];
    offset_t osy = stride_out[ndim-3];
    offset_t osz = stride_out[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        reduce_t z = static_cast<reduce_t>(grid[grid_offset + gsc * 2]);
        if (!InFOV<extrapolate, three>::infov(x, y, z, nx, ny, nz))
            continue;
        offset_t inp_offset = index2offset(i, ndim-1, size_grid, stride_inp);
        offset_t out_offset = index2offset(i, ndim-4, size_grid, stride_out);

        PushPull<three, IX, BX, IY, BY, IZ, BZ>::push(
            out + out_offset, inp + inp_offset,
            x, nx, osx, y, ny, osy, z, nz, osz, nc, osc, isc);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void count3d(scalar_t * out, const scalar_t * grid, int ndim,
             const offset_t * size_grid,
             const offset_t * size_splinc,
             const offset_t * stride_out,
             const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-4];
    offset_t ny  = size_splinc[ndim-3];
    offset_t nz  = size_splinc[ndim-2];
    offset_t osx = stride_out[ndim-4];
    offset_t osy = stride_out[ndim-3];
    offset_t osz = stride_out[ndim-2];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        reduce_t z = static_cast<reduce_t>(grid[grid_offset + gsc * 2]);
        if (!InFOV<extrapolate, three>::infov(x, y, z, nx, ny, nz))
            continue;
        offset_t out_offset = index2offset(i, ndim-4, size_grid, stride_out);

        PushPull<three, IX, BX, IY, BY, IZ, BZ>::count(
            out + out_offset, x, nx, osx, y, ny, osy, z, nz, osz);
    }
}


template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void grad3d(scalar_t * out,
            const scalar_t * inp, const scalar_t * grid, int ndim,
            const offset_t * size_grid,
            const offset_t * size_splinc,
            const offset_t * stride_out,
            const offset_t * stride_inp,
            const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-4];
    offset_t ny  = size_splinc[ndim-3];
    offset_t nz  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
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
        reduce_t z = static_cast<reduce_t>(grid[grid_offset + gsc * 2]);
        if (!InFOV<extrapolate, three>::infov(x, y, z, nx, ny, nz)) {
            for (offset_t c=0; c<nc; ++c) {
                out[out_offset + c * osc]           = static_cast<scalar_t>(0);
                out[out_offset + c * osc + osg]     = static_cast<scalar_t>(0);
                out[out_offset + c * osc + osg * 2] = static_cast<scalar_t>(0);
            }
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-4, size_grid, stride_inp);

        PushPull<three, IX, BX, IY, BY, IZ, BZ>::grad(
            out + out_offset, inp + inp_offset,
            x, nx, isx, y, ny, isy, z, nz, isz, nc, osc, isc, osg);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void pull3d_backward(
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

    offset_t nx  = size_splinc[ndim-4];
    offset_t ny  = size_splinc[ndim-3];
    offset_t nz  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-4];
    offset_t osy = stride_out[ndim-3];
    offset_t osz = stride_out[ndim-2];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
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
        reduce_t z = static_cast<reduce_t>(grid[grid_offset + gsc * 2]);
        if (!InFOV<extrapolate, three>::infov(x, y, z, nx, ny, nz)) {
            gout[gout_offset]           = static_cast<scalar_t>(0);
            gout[gout_offset + osg]     = static_cast<scalar_t>(0);
            gout[gout_offset + osg * 2] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-4, size_grid, stride_inp);
        offset_t out_offset = index2offset(i, ndim-4, size_grid, stride_out);
        offset_t ginp_offset = index2offset(i, ndim-1, size_grid, stride_ginp);

        PushPull<three, IX, BX, IY, BY, IZ, BZ>::pull_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, osx, isx, y, ny, osy, isy, z, nz, osz, isz,
            nc, osc, isc, osg, isg);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void push3d_backward(
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

    offset_t nx  = size_splinc[ndim-4];
    offset_t ny  = size_splinc[ndim-3];
    offset_t nz  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
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
        reduce_t z = static_cast<reduce_t>(grid[grid_offset + gsc * 2]);
        if (!InFOV<extrapolate, three>::infov(x, y, z, nx, ny, nz)) {
            for (offset_t c=0; c<nc; ++c)
                out[out_offset + c * osc] = static_cast<scalar_t>(0);
            gout[gout_offset]           = static_cast<scalar_t>(0);
            gout[gout_offset + osg]     = static_cast<scalar_t>(0);
            gout[gout_offset + osg * 2] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-1, size_grid, stride_inp);
        offset_t ginp_offset = index2offset(i, ndim-4, size_grid, stride_ginp);

        PushPull<three, IX, BX, IY, BY, IZ, BZ>::push_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, isx, y, ny, isy, z, nz, isz,
            nc, osc, isc, osg, isg);
    }
}


template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void count3d_backward(
    scalar_t * gout, const scalar_t * ginp,
    const scalar_t * grid, int ndim,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_gout,
    const offset_t * stride_ginp,
    const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-4];
    offset_t ny  = size_splinc[ndim-3];
    offset_t nz  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t sx  = stride_ginp[ndim-4];
    offset_t sy  = stride_ginp[ndim-3];
    offset_t sz  = stride_ginp[ndim-2];
    offset_t osg = stride_gout[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
        reduce_t z = static_cast<reduce_t>(grid[grid_offset + gsc * 2]);
        if (!InFOV<extrapolate, three>::infov(x, y, z, nx, ny, nz)) {
            gout[gout_offset]           = static_cast<scalar_t>(0);
            gout[gout_offset + osg]     = static_cast<scalar_t>(0);
            gout[gout_offset + osg * 2] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t ginp_offset = index2offset(i, ndim-4, size_grid, stride_ginp);

        PushPull<three, IX, BX, IY, BY, IZ, BZ>::count_backward(
            gout + gout_offset, ginp + ginp_offset,
            x, nx, sx, y, ny, sy, z, nz, sz, osg);
    }
}

template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t>
void grad3d_backward(
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

    offset_t nx  = size_splinc[ndim-4];
    offset_t ny  = size_splinc[ndim-3];
    offset_t nz  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-4];
    offset_t osy = stride_out[ndim-3];
    offset_t osz = stride_out[ndim-2];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
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
        reduce_t z = static_cast<reduce_t>(grid[grid_offset + grsc * 2]);
        if (!InFOV<extrapolate, three>::infov(x, y, z, nx, ny, nz)) {
            gout[gout_offset]           = static_cast<scalar_t>(0);
            gout[gout_offset + osg]     = static_cast<scalar_t>(0);
            gout[gout_offset + osg * 2] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-4, size_grid, stride_inp);
        offset_t out_offset = index2offset(i, ndim-4, size_grid, stride_out);
        offset_t ginp_offset = index2offset(i, ndim-1, size_grid, stride_ginp);

        PushPull<three, IX, BX, IY, BY>::grad_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            x, nx, osx, isx, y, ny, osy, isy, z, nz, osz, isz,
            nc, osc, isc, gsc, osg, isg);
    }
}

} // namespace pushpull
} // namespace jf

#endif // JF_PUSHPULL_LOOP3D
