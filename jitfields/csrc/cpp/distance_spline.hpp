#ifndef JF_DISTANCE_SPLINE_LOOP
#define JF_DISTANCE_SPLINE_LOOP
#include "../lib/cuda_switch.h"
#include "../lib/distance.h"
#include "../lib/batch.h"
#include "../lib/parallel.h"

namespace jf {
namespace distance_spline {


// Compute the minimum distance from a set of points to a 1D spline
// using a dictionary of times
template <
    int nbatch,             // Number of batch dimensions
    int ndim,               // Number of spatial dimensions
    spline::type S,         // Spline order
    bound::type B,          // Boundary condition
    typename scalar_t,      // Value data type
    typename offset_t       // Index/Stride data type
>
void mindist_table(
    scalar_t * time,                // (*batch) tensor -> Best time
    scalar_t * dist,                // (*batch) tensor -> Best sqdist
    const scalar_t * loc,           // (*batch, ndim) tensor -> ND location of each point
    const scalar_t * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
    const scalar_t * times,         // (*batch) tensor -> Time values to try
    offset_t ntimes,                // Number of times values to try
    const offset_t * _size,         // [*batch, npoints, ndim] list -> Coeff shape
    const offset_t * _stride_time,  // [*batch] list -> Strides of `time`
    const offset_t * _stride_dist,  // [*batch] list -> Strides of `dist`
    const offset_t * _stride_loc,   // [*batch, ndim] list -> Strides of `loc`
    const offset_t * _stride_coeff, // [*batch, npoints, ndim] list -> Strides or `coeff`
    const offset_t * _stride_times  // [*batch, ntimes] list -> Strides of `times`
)
{
    using Klass = SplineDist<ndim, S, B, scalar_t, offset_t>;

    offset_t size         [nbatch+2]; fillfrom<nbatch+2> (size,         _size);
    offset_t stride_time  [nbatch];   fillfrom<nbatch>   (stride_time,  _stride_time);
    offset_t stride_dist  [nbatch];   fillfrom<nbatch>   (stride_dist,  _stride_dist);
    offset_t stride_loc   [nbatch+1]; fillfrom<nbatch+1> (stride_loc,   _stride_loc);
    offset_t stride_coeff [nbatch+2]; fillfrom<nbatch+2> (stride_coeff, _stride_coeff);
    offset_t stride_times [nbatch+1]; fillfrom<nbatch+1> (stride_times, _stride_times);

    offset_t numel = prod<nbatch>(size);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_time  = index2offset<nbatch>(i, size, stride_time);
        offset_t offset_dist  = index2offset<nbatch>(i, size, stride_dist);
        offset_t offset_loc   = index2offset<nbatch>(i, size, stride_loc);
        offset_t offset_coeff = index2offset<nbatch>(i, size, stride_coeff);
        offset_t offset_times = index2offset<nbatch>(i, size, stride_times);

        Klass::min_table(
            time + offset_time,
            dist + offset_dist,
            loc + offset_loc, 
            coeff + offset_coeff, 
            times + offset_times, 
            ntimes,
            stride_times[nbatch],
            stride_loc[nbatch], 
            size[nbatch],
            stride_coeff[nbatch], 
            stride_coeff[nbatch+1]
        );
    }});
}

// Compute the minimum distance from a set of points to a 1D spline
// using Brent (gradient-free) optimization
template <
    int nbatch,             // Number of batch dimensions
    int ndim,               // Number of spatial dimensions
    spline::type S,         // Spline order
    bound::type B,          // Boundary condition
    typename scalar_t,      // Value data type
    typename offset_t       // Index/Stride data type
>
void mindist_brent(
    scalar_t * time,                // (*batch) tensor -> Best time
    scalar_t * dist,                // (*batch) tensor -> Best sqdist
    const scalar_t * loc,           // (*batch, ndim) tensor -> ND location of each point
    const scalar_t * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
    const offset_t * _size,         // [*batch, npoints, ndim] list -> Coeff shape
    const offset_t * _stride_time,  // [*batch] list -> Strides of `time`
    const offset_t * _stride_dist,  // [*batch] list -> Strides of `dist`
    const offset_t * _stride_loc,   // [*batch, ndim] list -> Strides of `loc`
    const offset_t * _stride_coeff, // [*batch, npoints, ndim] list -> Strides or `coeff`
    offset_t max_iter,
    scalar_t tol,
    scalar_t step
)
{
    using Klass = SplineDist<ndim, S, B, scalar_t, offset_t>;

    offset_t size         [nbatch+2]; fillfrom<nbatch+2> (size,         _size);
    offset_t stride_time  [nbatch];   fillfrom<nbatch>   (stride_time,  _stride_time);
    offset_t stride_dist  [nbatch];   fillfrom<nbatch>   (stride_dist,  _stride_dist);
    offset_t stride_loc   [nbatch+1]; fillfrom<nbatch+1> (stride_loc,   _stride_loc);
    offset_t stride_coeff [nbatch+2]; fillfrom<nbatch+2> (stride_coeff, _stride_coeff);

    offset_t numel = prod<nbatch>(size);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_time  = index2offset<nbatch>(i, size, stride_time);
        offset_t offset_dist  = index2offset<nbatch>(i, size, stride_dist);
        offset_t offset_loc   = index2offset<nbatch>(i, size, stride_loc);
        offset_t offset_coeff = index2offset<nbatch>(i, size, stride_coeff);    

        Klass::min_brent(
            time + offset_time,
            dist + offset_dist,
            loc + offset_loc, 
            coeff + offset_coeff, 
            stride_loc[nbatch], 
            size[nbatch],
            stride_coeff[nbatch], 
            stride_coeff[nbatch+1], 
            max_iter,
            tol,
            step
        );
    }});
}


// Compute the minimum distance from a set of points to a 1D spline
// using Gauss-Newton optimization
template <
    int nbatch,             // Number of batch dimensions
    int ndim,               // Number of spatial dimensions
    spline::type S,         // Spline order
    bound::type B,          // Boundary condition
    typename scalar_t,      // Value data type
    typename offset_t       // Index/Stride data type
>
void mindist_gaussnewton(
    scalar_t * time,                // (*batch) tensor -> Best time
    scalar_t * dist,                // (*batch) tensor -> Best sqdist
    const scalar_t * loc,           // (*batch, ndim) tensor -> ND location of each point
    const scalar_t * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
    const offset_t * _size,         // [*batch, npoints, ndim] list -> Coeff shape
    const offset_t * _stride_time,  // [*batch] list -> Strides of `time`
    const offset_t * _stride_dist,  // [*batch] list -> Strides of `dist`
    const offset_t * _stride_loc,   // [*batch, ndim] list -> Strides of `loc`
    const offset_t * _stride_coeff, // [*batch, npoints, ndim] list -> Strides or `coeff`
    offset_t max_iter,
    scalar_t tol
)
{
    using Klass = SplineDist<ndim, S, B, scalar_t, offset_t>;

    offset_t size         [nbatch+2]; fillfrom<nbatch+2> (size,         _size);
    offset_t stride_time  [nbatch];   fillfrom<nbatch>   (stride_time,  _stride_time);
    offset_t stride_dist  [nbatch];   fillfrom<nbatch>   (stride_dist,  _stride_dist);
    offset_t stride_loc   [nbatch+1]; fillfrom<nbatch+1> (stride_loc,   _stride_loc);
    offset_t stride_coeff [nbatch+2]; fillfrom<nbatch+2> (stride_coeff, _stride_coeff);

    offset_t numel = prod<nbatch>(size);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_time  = index2offset<nbatch>(i, size, stride_time);
        offset_t offset_dist  = index2offset<nbatch>(i, size, stride_dist);
        offset_t offset_loc   = index2offset<nbatch>(i, size, stride_loc);
        offset_t offset_coeff = index2offset<nbatch>(i, size, stride_coeff);       

        Klass::min_gaussnewton(
            time + offset_time,
            dist + offset_dist,
            loc + offset_loc, 
            coeff + offset_coeff, 
            stride_loc[nbatch], 
            size[nbatch],
            stride_coeff[nbatch], 
            stride_coeff[nbatch+1], 
            max_iter,
            tol
        );
    }});
}

} // namespace distance_spline
} // namespace jf

#endif // JF_DISTANCE_SPLINE_LOOP
