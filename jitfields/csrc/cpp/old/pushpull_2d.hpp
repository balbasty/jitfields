#ifndef JF_PUSHPULL_LOOP2D
#define JF_PUSHPULL_LOOP2D
#include "../lib/cuda_switch.h"
#include "../lib/pushpull.h"
#include "../lib/batch.h"
#include "../lib/parallel.h"

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

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset(i, ndim-1, size_grid, stride_out);
            offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
            reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
            reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
            if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
                for (offset_t c=0; c<nc; ++c)
                    out[out_offset + c * osc] = static_cast<scalar_t>(0);
                continue;
            }
            offset_t inp_offset = index2offset(i, ndim-3, size_grid, stride_inp);

            PushPull<two, IX, BX, IY, BY>::pull(
                out + out_offset, inp + inp_offset,
                x, nx, isx, y, ny, isy, nc, osc, isc);
        }
    });
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
    int nbatch = ndim - 3;
    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t osx = stride_out[ndim-3];
    offset_t osy = stride_out[ndim-2];
    offset_t osc = stride_out[ndim-1];
    offset_t isc = stride_inp[ndim-1];
    offset_t gsc = stride_grid[ndim-1];

    if ( jf::has_atomic_add<scalar_t>::value )
    {
        offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels
        parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
            for (offset_t i=start; i < end; ++i)
            {
                offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
                reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
                reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
                if (!InFOV<extrapolate, two>::infov(x, y, nx, ny))
                    continue;
                offset_t inp_offset = index2offset(i, ndim-1, size_grid, stride_inp);
                offset_t out_offset = index2offset(i, nbatch, size_grid, stride_out);

                PushPull<two, IX, BX, IY, BY>::push(
                    out + out_offset, inp + inp_offset,
                    x, nx, osx, y, ny, osy, nc, osc, isc);
            }
        });
    }
    else
    {
        offset_t numel_batch   = prod(size_grid, nbatch);
        offset_t numel_spatial = prod(size_grid+nbatch, 2);
        long grain_size = max(GRAIN_SIZE/numel_spatial, 1L);
        parallel_for(0, numel_batch, grain_size, [&](long start, long end) {
            for (offset_t i=start; i < end; ++i)
            {
                offset_t grid_offset0 = index2offset(i, nbatch, size_grid, stride_grid);
                offset_t inp_offset0  = index2offset(i, nbatch, size_grid, stride_inp);
                offset_t out_offset   = index2offset(i, nbatch, size_grid, stride_out);
                for (offset_t j=0; j < numel_spatial; ++j)
                {
                    offset_t grid_offset = grid_offset0
                                         + index2offset(j, 2, size_grid+nbatch, stride_grid+nbatch);
                    reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
                    reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
                    if (!InFOV<extrapolate, two>::infov(x, y, nx, ny))
                        continue;
                    offset_t inp_offset = inp_offset0
                                        + index2offset(j, 2, size_grid+nbatch, stride_inp+nbatch);

                    PushPull<two, IX, BX, IY, BY>::push(
                        out + out_offset, inp + inp_offset,
                        x, nx, osx, y, ny, osy, nc, osc, isc);
                }
            }
        });
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
    int nbatch = ndim - 3;
    offset_t nx  = size_splinc[ndim-3];
    offset_t ny  = size_splinc[ndim-2];
    offset_t osx = stride_out[ndim-3];
    offset_t osy = stride_out[ndim-2];
    offset_t gsc = stride_grid[ndim-1];

    if ( jf::has_atomic_add<scalar_t>::value )
    {
        offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels
        parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
            for (offset_t i=start; i < end; ++i)
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
        });
    }
    else
    {
        offset_t numel_batch   = prod(size_grid, nbatch);
        offset_t numel_spatial = prod(size_grid+nbatch, 2);
        long grain_size = max(GRAIN_SIZE/numel_spatial, 1L);
        parallel_for(0, numel_batch, grain_size, [&](long start, long end) {
            for (offset_t i=start; i < end; ++i)
            {
                offset_t grid_offset0 = index2offset(i, nbatch, size_grid, stride_grid);
                offset_t out_offset   = index2offset(i, nbatch, size_grid, stride_out);
                for (offset_t j=0; j < numel_spatial; ++j)
                {
                    offset_t grid_offset = grid_offset0
                                         + index2offset(j, 2, size_grid+nbatch, stride_grid+nbatch);
                    reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
                    reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
                    if (!InFOV<extrapolate, two>::infov(x, y, nx, ny))
                        continue;

                    PushPull<two, IX, BX, IY, BY>::count(
                        out + out_offset, x, nx, osx, y, ny, osy);
                }
            }
        });
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

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset(i, ndim-1, size_grid, stride_out);
            offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
            reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
            reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
            if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
                for (offset_t c=0; c<nc; ++c) {
                    out[out_offset + c * osc]       = static_cast<scalar_t>(0);
                    out[out_offset + c * osc + osg] = static_cast<scalar_t>(0);
                }
                continue;
            }
            offset_t inp_offset = index2offset(i, ndim-3, size_grid, stride_inp);

            PushPull<two, IX, BX, IY, BY>::grad(
                out + out_offset, inp + inp_offset,
                x, nx, isx, y, ny, isy, nc, osc, isc, osg);
        }
    });
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
    int nbatch = ndim - 3;
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

    if ( jf::has_atomic_add<scalar_t>::value )
    {
        offset_t numel = prod(size_grid, ndim-1);
        parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
            for (offset_t i=start; i < end; ++i)
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
        });
    }
    else
    {
        const offset_t * size_grid1   = size_grid   + nbatch;
        const offset_t * stride_grid1 = stride_grid + nbatch;
        const offset_t * stride_gout1 = stride_gout + nbatch;
        const offset_t * stride_ginp1 = stride_ginp + nbatch;
        offset_t numel_batch   = prod(size_grid, nbatch);
        offset_t numel_spatial = prod(size_grid1, 2);
        long grain_size = max(GRAIN_SIZE/numel_spatial, 1L);
        parallel_for(0, numel_batch, grain_size, [&](long start, long end) {
            for (offset_t i=start; i < end; ++i)
            {
                offset_t grid_offset0 = index2offset(i, nbatch, size_grid, stride_grid);
                offset_t gout_offset0 = index2offset(i, nbatch, size_grid, stride_gout);
                offset_t inp_offset   = index2offset(i, nbatch, size_grid, stride_inp);
                offset_t out_offset   = index2offset(i, nbatch, size_grid, stride_out);
                offset_t ginp_offset0 = index2offset(i, nbatch, size_grid, stride_ginp);
                offset_t grid_offset, gout_offset, ginp_offset;
                for (offset_t j=0; j < numel_spatial; ++j) {
                    grid_offset = grid_offset0 + index2offset(j, 2, size_grid1, stride_grid1);
                    gout_offset = gout_offset0 + index2offset(j, 2, size_grid1, stride_gout1);
                    reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
                    reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
                    if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
                        gout[gout_offset]       = static_cast<scalar_t>(0);
                        gout[gout_offset + osg] = static_cast<scalar_t>(0);
                        continue;
                    }
                    ginp_offset = ginp_offset0 + index2offset(j, 2, size_grid1, stride_ginp1);

                    PushPull<two, IX, BX, IY, BY>::pull_backward(
                        out + out_offset, gout + gout_offset,
                        inp + inp_offset, ginp + ginp_offset,
                        x, nx, osx, isx, y, ny, osy, isy, nc, osc, isc, osg, isg);
                }
            }
        });
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

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
            offset_t out_offset  = index2offset(i, ndim-1, size_grid, stride_out);
            offset_t gout_offset = index2offset(i, ndim-1, size_grid, stride_gout);
            reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
            reduce_t y = static_cast<reduce_t>(grid[grid_offset + gsc]);
            if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
                for (offset_t c=0; c<nc; ++c)
                    out[out_offset + c * osc] = static_cast<scalar_t>(0);
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
    });
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

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
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
    });
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
    int nbatch = ndim - 3;
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

    if ( jf::has_atomic_add<scalar_t>::value )
    {
        offset_t numel = prod(size_grid, ndim-1);
        parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
            for (offset_t i=start; i < end; ++i)
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
        });
    }
    else
    {
        const offset_t * size_grid1   = size_grid   + nbatch;
        const offset_t * stride_grid1 = stride_grid + nbatch;
        const offset_t * stride_gout1 = stride_gout + nbatch;
        const offset_t * stride_ginp1 = stride_ginp + nbatch;
        offset_t numel_batch   = prod(size_grid, nbatch);
        offset_t numel_spatial = prod(size_grid+nbatch, 2);
        long grain_size = max(GRAIN_SIZE/numel_spatial, 1L);
        parallel_for(0, numel_batch, grain_size, [&](long start, long end) {
            for (offset_t i=start; i < end; ++i)
            {
                offset_t grid_offset0 = index2offset(i, nbatch, size_grid, stride_grid);
                offset_t gout_offset0 = index2offset(i, nbatch, size_grid, stride_gout);
                offset_t inp_offset   = index2offset(i, nbatch, size_grid, stride_inp);
                offset_t out_offset   = index2offset(i, nbatch, size_grid, stride_out);
                offset_t ginp_offset0 = index2offset(i, nbatch, size_grid, stride_ginp);
                offset_t grid_offset, gout_offset, ginp_offset;
                for (offset_t j=0; j < numel_spatial; ++j)
                {
                    grid_offset = grid_offset0 + index2offset(j, 2, size_grid1, stride_grid1);
                    gout_offset = gout_offset0 + index2offset(j, 2, size_grid1, stride_gout1);
                    reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
                    reduce_t y = static_cast<reduce_t>(grid[grid_offset + grsc]);
                    if (!InFOV<extrapolate, two>::infov(x, y, nx, ny)) {
                        gout[gout_offset]       = static_cast<scalar_t>(0);
                        gout[gout_offset + osg] = static_cast<scalar_t>(0);
                        continue;
                    }
                    ginp_offset = ginp_offset0 + index2offset(j, 2, size_grid1, stride_ginp1);

                    PushPull<two, IX, BX, IY, BY>::grad_backward(
                        out + out_offset, gout + gout_offset,
                        inp + inp_offset, ginp + ginp_offset,
                        x, nx, osx, isx, y, ny, osy, isy, nc, osc, isc, gsc, osg, isg);
                }
            }
        });
    }
}

} // namespace pushpull
} // namespace jf

#endif // JF_PUSHPULL_LOOP2D
