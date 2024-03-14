#include "../lib/cuda_switch.h"
#include "../lib/spline.h"
#include "../lib/bounds.h"
#include "../lib/batch.h"
#include "../lib/pushpull.h"

using namespace std;
using namespace jf;
using namespace jf::pushpull;

template <int nbatch, int ndim, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
__global__
void pull(
    scalar_t * out,                // (*batch, *spatial_grid, C) tensor | Placeholder for the pulled volume
    const scalar_t * inp,          // (*batch, *spatial_spln, C) tensor | Input volume
    const scalar_t * grid,         // (*batch, *spatial_grid, D) tensor | Coordinates into the input volume
    const offset_t * _size_grid,   // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc, // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,  // [*batch, *spatial_grid, C] vector
    const offset_t * _stride_inp,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_grid) // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t gsc = stride_grid[nall];

    auto pull = [&](const reduce_t * loc, offset_t out_offset, offset_t inp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ>::pull(
            out + out_offset, inp + inp_offset,
            loc, size_splinc + nbatch, stride_inp + nbatch, nc, osc, isc);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nall>(i, size_grid, stride_out);
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);

        reduce_t loc[ndim]; fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!InFOV<extrapolate, ndim>::infov(loc, size_splinc+nbatch))
        {
            for (offset_t c=0; c<nc; ++c)
                out[out_offset + c * osc] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset = index2offset<nbatch>(i, size_grid, stride_inp);

        pull(loc, out_offset, inp_offset);
    }
}

template <int nbatch, int ndim, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
__global__
void push(
    scalar_t * out,                   // (*batch, *spatial_spln, C) tensor | Placeholder for the splatted volume
    const scalar_t * inp,             // (*batch, *spatial_grid, C) tensor | Input volume
    const scalar_t * grid,            // (*batch, *spatial_grid, D) tensor | Coordinates into the output volume
    const offset_t * _size_grid,      // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,    // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,     // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_inp,     // [*batch, *spatial_grid, C] vector
    const offset_t * _stride_grid)    // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t gsc = stride_grid[nall];

    auto push = [&] (const reduce_t * loc, offset_t out_offset, offset_t inp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ>::push(
            out + out_offset, inp + inp_offset,
            loc, size_splinc + nbatch, stride_out + nbatch, nc, osc, isc);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);

        reduce_t loc[ndim]; fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!InFOV<extrapolate, ndim>::infov(loc, _size_splinc+nbatch))
            continue;

        offset_t inp_offset = index2offset<nall>(i, size_grid, stride_inp);
        offset_t out_offset = index2offset<nbatch>(i, size_grid, stride_out);

        push(loc, out_offset, inp_offset);
    }
}

template <int nbatch, int ndim, int extrapolate,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
__global__
void count(
    scalar_t * out,                  // (*batch, *spatial_spln, C) tensor | Placeholder for the count image
    const scalar_t * grid,           // (*batch, *spatial_grid, D) tensor | Coordinates into the output volume
    const offset_t * _size_grid,     // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,    // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_grid)   // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t gsc = stride_grid[nall];

    auto count = [&](const reduce_t * loc, offset_t out_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ>::count(
            out + out_offset, loc, size_splinc + nbatch, stride_out + nbatch);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);

        reduce_t loc[ndim]; fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!InFOV<extrapolate, ndim>::infov(loc, size_splinc+nbatch))
            continue;

        offset_t out_offset = index2offset<nbatch>(i, size_grid, stride_out);

        count(loc, out_offset);
    }
}

template <int nbatch, int ndim, int extrapolate, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
__global__
void grad(
    scalar_t * out,                 // (*batch, *spatial_grid, C, D) tensor | Placeholder for the pulled gradients
    const scalar_t * inp,           // (*batch, *spatial_spln, C) tensor    | Input volume
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor    | Coordinates into the input volume
    const offset_t * _size_grid,    // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial_grid, C, D] vector
    const offset_t * _stride_inp,   // [*batch, *spatial_spln, C] vector
            const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+2]; fillfrom<nall+2>(stride_out,  _stride_out);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t osg = stride_out[nall+1];
    offset_t isc = stride_inp[nall];
    offset_t gsc = stride_grid[nall];

    auto grad = [&](const reduce_t * loc, offset_t out_offset, offset_t inp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::grad(
            out + out_offset, inp + inp_offset,
            loc, size_splinc + nbatch, stride_inp + nbatch,
            nc, osc, isc, osg);
    };

    offset_t numel = prod<nall>(size_grid);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nall>(i, size_grid, stride_out);
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!InFOV<extrapolate, ndim>::infov(loc, size_splinc + nbatch))
        {
            for (offset_t c=0; c<nc; ++c)
                fill<ndim>(out + out_offset + c * osc, 0, osg);
            continue;
        }

        offset_t inp_offset = index2offset<nbatch>(i, size_grid, stride_inp);

        grad(loc, out_offset, inp_offset);
    }
}

template <int nbatch, int ndim, int extrapolate, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
__global__
void hess(
    scalar_t * out,                 // (*batch, *spatial_grid, C, D) tensor | Placeholder for the pulled gradients
    const scalar_t * inp,           // (*batch, *spatial_spln, C) tensor    | Input volume
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor    | Coordinates into the input volume
    const offset_t * _size_grid,    // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial_grid, C, D] vector
    const offset_t * _stride_inp,   // [*batch, *spatial_spln, C] vector
            const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+2]; fillfrom<nall+2>(stride_out,  _stride_out);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t osg = stride_out[nall+1];
    offset_t isc = stride_inp[nall];
    offset_t gsc = stride_grid[nall];

    auto grad = [&](const reduce_t * loc, offset_t out_offset, offset_t inp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::hess(
            out + out_offset, inp + inp_offset,
            loc, size_splinc + nbatch, stride_inp + nbatch,
            nc, osc, isc, osg);
    };

    offset_t numel = prod<nall>(size_grid);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nall>(i, size_grid, stride_out);
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!InFOV<extrapolate, ndim>::infov(loc, size_splinc + nbatch))
        {
            for (offset_t c=0; c<nc; ++c)
                fill<(ndim*(ndim+1))/2>(out + out_offset + c * osc, 0, osg);
            continue;
        }

        offset_t inp_offset = index2offset<nbatch>(i, size_grid, stride_inp);

        hess(loc, out_offset, inp_offset);
    }
}

template <int nbatch, int ndim, int extrapolate, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
__global__
void pull_backward(
    scalar_t * out,                 // (*batch, *spatial_spln, C) tensor | Placeholder for the gradient wrt `inp`
    scalar_t * gout,                // (*batch, *spatial_grid, D) tensor | Placeholder for the gradient wrt `grid`
    const scalar_t * inp,           // (*batch, *spatial_spln, C) tensor | Input volume of the forward pass
    const scalar_t * ginp,          // (*batch, *spatial_grid, C) tensor | Gradient wrt to the output of the forward pass
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor | Coordinates into the input volume
    const offset_t * _size_grid,    // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_gout,  // [*batch, *spatial_grid, D] vector
    const offset_t * _stride_inp,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_ginp,  // [*batch, *spatial_grid, C] vector
    const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_gout [nall+1]; fillfrom<nall+1>(stride_gout, _stride_gout);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_ginp [nall+1]; fillfrom<nall+1>(stride_ginp, _stride_ginp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t isg = stride_ginp[nall];
    offset_t osg = stride_gout[nall];
    offset_t gsc = stride_grid[nall];

    auto pull_backward = [&](
        const reduce_t * loc,
        offset_t out_offset,
        offset_t gout_offset,
        offset_t inp_offset,
        offset_t ginp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::pull_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            loc, size_splinc + nbatch,
            stride_out + nbatch, stride_inp + nbatch,
            nc, osc, isc, osg, isg);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);
        offset_t gout_offset = index2offset<nall>(i, size_grid, stride_gout);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!InFOV<extrapolate, ndim>::infov(loc, size_splinc + nbatch))
        {
            fill<ndim>(gout + gout_offset, 0, osg);
            continue;
        }

        offset_t inp_offset  = index2offset<nbatch>(i, size_grid, stride_inp);
        offset_t out_offset  = index2offset<nbatch>(i, size_grid, stride_out);
        offset_t ginp_offset = index2offset<nall>(i, size_grid, stride_ginp);

        pull_backward(loc, out_offset, gout_offset, inp_offset, ginp_offset);
    }
}

template <int nbatch, int ndim, int extrapolate, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
__global__
void push_backward(
    scalar_t * out,                 // (*batch, *spatial_grid, C) tensor | Placeholder for the gradient wrt `inp`
    scalar_t * gout,                // (*batch, *spatial_grid, D) tensor | Placeholder for the gradient wrt `grid`
    const scalar_t * inp,           // (*batch, *spatial_grid, C) tensor | Input volume of the forward pass
    const scalar_t * ginp,          // (*batch, *spatial_spln, C) tensor | Gradient wrt the output of the forward pass
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor | Coordinates into the output of the forward pass
    const offset_t * _size_grid,    // [*batch, *spatial_spln, C] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_gout,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_ginp,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_gout [nall+1]; fillfrom<nall+1>(stride_gout,  _stride_gout);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_ginp [nall+1]; fillfrom<nall+1>(stride_ginp, _stride_ginp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t isg = stride_ginp[nall];
    offset_t osg = stride_gout[nall];
    offset_t gsc = stride_grid[nall];

    auto push_backward = [&](
        const reduce_t * loc,
        offset_t out_offset,
        offset_t gout_offset,
        offset_t inp_offset,
        offset_t ginp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::push_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            loc, size_splinc + nbatch, stride_inp + nbatch,
            nc, osc, isc, osg, isg);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);
        offset_t out_offset  = index2offset<nall>(i, size_grid, stride_out);
        offset_t gout_offset = index2offset<nall>(i, size_grid, stride_gout);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!InFOV<extrapolate, ndim>::infov(loc, size_splinc+nbatch))
        {
            for (offset_t c=0; c<nc; ++c)
                out[out_offset + c * osc] = static_cast<scalar_t>(0);
            fill<ndim>(gout + gout_offset, 0, osg);
            continue;
        }

        offset_t inp_offset = index2offset<nall>(i, size_grid, stride_inp);
        offset_t ginp_offset = index2offset<nbatch>(i, size_grid, stride_ginp);
        push_backward(loc, out_offset, gout_offset, inp_offset, ginp_offset);
    }
}


template <int nbatch, int ndim, int extrapolate, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
__global__
void count_backward(
    scalar_t * gout,                // (*batch, *spatial_grid, D) tensor | Placeholder for the gradient wrt `grid`
    const scalar_t * ginp,          // (*batch, *spatial_spln, 1) tensor | Gradient wrt to the output of the forward pass
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor | Coordinates into the output of the forward pass
    const offset_t * _size_grid,    // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, 1] vector
    const offset_t * _stride_gout,  // [*batch, *spatial_grid, D] vector
    const offset_t * _stride_ginp,  // [*batch, *spatial_spln, 1] vector
    const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_gout [nall+1]; fillfrom<nall+1>(stride_gout, _stride_gout);
    offset_t stride_ginp [nall+1]; fillfrom<nall+1>(stride_ginp, _stride_ginp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osg = stride_gout[nall];
    offset_t gsc = stride_grid[nall];

    auto count_backward = [&](
        const reduce_t * loc,
        offset_t gout_offset,
        offset_t ginp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::count_backward(
            gout + gout_offset, ginp + ginp_offset,
            loc, size_splinc + nbatch, stride_ginp + nbatch, osg);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);
        offset_t gout_offset = index2offset<nall>(i, size_grid, stride_gout);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!InFOV<extrapolate, ndim>::infov(loc, size_splinc + nbatch))
        {
            fill<ndim>(gout + gout_offset, 0, osg);
            continue;
        }

        offset_t ginp_offset = index2offset<nbatch>(i, size_grid, stride_ginp);
        count_backward(loc, gout_offset, ginp_offset);
    }
}

template <int nbatch, int ndim, int extrapolate, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
__global__
void grad_backward(
    scalar_t * out,                 // (*batch, *spatial_spln, C) tensor    | Placeholder for the gradient wrt `inp`
    scalar_t * gout,                // (*batch, *spatial_grid, D) tensor    | Placeholder for the gradient wrt `grid`
    const scalar_t * inp,           // (*batch, *spatial_spln, C) tensor    | Input of the forward pass
    const scalar_t * ginp,          // (*batch, *spatial_grid, C, D) tensor | Gradient wrt the output of the forward pass
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor    | Coordinates into the input volume
    const offset_t * _size_grid,    // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_gout,  // [*batch, *spatial_grid, D] vector
    const offset_t * _stride_inp,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_ginp,  // [*batch, *spatial_grid, C, D] vector
    const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_gout [nall+1]; fillfrom<nall+1>(stride_gout, _stride_gout);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_ginp [nall+2]; fillfrom<nall+2>(stride_ginp, _stride_ginp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc   = size_splinc[nall];
    offset_t osc  = stride_out[nall];
    offset_t isc  = stride_inp[nall];
    offset_t isg  = stride_ginp[nall+1];
    offset_t gsc  = stride_ginp[nall];
    offset_t osg  = stride_gout[nall];
    offset_t grsc = stride_grid[nall];

    auto grad_backward = [&](
        const reduce_t * loc,
        offset_t out_offset,
        offset_t gout_offset,
        offset_t inp_offset,
        offset_t ginp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::grad_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            loc, size_splinc + nbatch,
            stride_out + nbatch, stride_inp + nbatch,
            nc, osc, isc, gsc, osg, isg);
    };

    auto get_grid_offset = [&](offset_t i) {
        return index2offset<nall>(i, size_grid, stride_grid); };
    auto get_gout_offset = [&](offset_t i) {
        return index2offset<nall>(i, size_grid, stride_gout); };
    auto get_inp_offset = [&](offset_t i) {
        return index2offset<nbatch>(i, size_grid, stride_inp); };
    auto get_out_offset = [&](offset_t i) {
        return index2offset<nbatch>(i, size_grid, stride_out); };
    auto get_ginp_offset = [&](offset_t i) {
        return index2offset<nall>(i, size_grid, stride_ginp); };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = get_grid_offset(i);
        offset_t gout_offset = get_gout_offset(i);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, grsc);
        if (!InFOV<extrapolate, ndim>::infov(loc, size_splinc + nbatch))
        {
            fill<ndim>(gout + gout_offset, 0, osg);
            continue;
        }

        grad_backward(loc, get_out_offset(i), gout_offset,
                      get_inp_offset(i), get_ginp_offset(i));
    }
}
