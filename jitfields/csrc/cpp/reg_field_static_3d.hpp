#ifndef JF_REGULARISERS_FIELD_S_3D_LOOP
#define JF_REGULARISERS_FIELD_S_3D_LOOP
#include "../lib/cuda_switch.h"
#include "../lib/bounds.h"
#include "../lib/utils.h"
#include "../lib/batch.h"
#include "../lib/regularisers/field/3d_static.h"

namespace jf {
namespace reg_field {
namespace stat {

//======================================================================
//                              ABSOLUTE
//======================================================================

// --- ABSOLUTE: vel2mom -----------------------------------------------

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void vel2mom_absolute_3d(
    scalar_t * out, const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    const reduce_t absolute[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    // move kernel to the stack
    reduce_t kernel[C];
    for (offset_t c=0; c<C; ++c) kernel[c] = absolute[c];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t inp_offset = index2offset<ndim-1>(i, size, stride_inp);
            offset_t out_offset = index2offset<ndim-1>(i, size, stride_out);

            RegFieldStatic<C, three, BX, BY, BZ>::vel2mom_absolute(
                out + out_offset, inp + inp_offset, osc, isc, kernel);
        }
    });
}

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void add_vel2mom_absolute_3d(
    scalar_t * out, const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    const reduce_t absolute[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    // move kernel to the stack
    reduce_t kernel[C];
    for (offset_t c=0; c<C; ++c) kernel[c] = absolute[c];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t inp_offset = index2offset<ndim-1>(i, size, stride_inp);
            offset_t out_offset = index2offset<ndim-1>(i, size, stride_out);

            RegFieldStatic<C, three, BX, BY, BZ>::add_vel2mom_absolute(
                out + out_offset, inp + inp_offset, osc, isc, kernel);
        }
    });
}

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void sub_vel2mom_absolute_3d(
    scalar_t * out, const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    const reduce_t absolute[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    // move kernel to the stack
    reduce_t kernel[C];
    for (offset_t c=0; c<C; ++c) kernel[c] = absolute[c];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t inp_offset = index2offset<ndim-1>(i, size, stride_inp);
            offset_t out_offset = index2offset<ndim-1>(i, size, stride_out);

            RegFieldStatic<C, three, BX, BY, BZ>::sub_vel2mom_absolute(
                out + out_offset, inp + inp_offset, osc, isc, kernel);
        }
    });
}

// --- ABSOLUTE: kernel ------------------------------------------------

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void kernel_absolute_3d(
    scalar_t * out,
    const offset_t * size,
    const offset_t * stride,
    const reduce_t absolute[C])
{
    offset_t numel = prod<ndim-4>(size);  // loop across batch only

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t sx  = stride[ndim-4];
    offset_t sy  = stride[ndim-3];
    offset_t sz  = stride[ndim-2];
    offset_t sc  = stride[ndim-1];

    // move kernel to the stack
    reduce_t kernel[C];
    for (offset_t c=0; c<C; ++c) kernel[c] = absolute[c];

    offset_t offset = (nx-1)/2 * sx + (ny-1)/2 * sy + (nz-1)/2 * sz;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset<ndim-4>(i, size, stride);
            out_offset += offset;

            RegFieldStatic<C, three, BX, BY, BZ>::kernel_absolute(
                out + out_offset, sc, kernel);
        }
    });
}


template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void add_kernel_absolute_3d(
    scalar_t * out,
    const offset_t * size,
    const offset_t * stride,
    const reduce_t absolute[C])
{
    offset_t numel = prod<ndim-4>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t sx  = stride[ndim-4];
    offset_t sy  = stride[ndim-3];
    offset_t sz  = stride[ndim-2];
    offset_t sc  = stride[ndim-1];

    // move kernel to the stack
    reduce_t kernel[C];
    for (offset_t c=0; c<C; ++c) kernel[c] = absolute[c];

    offset_t offset = (nx-1)/2 * sx + (ny-1)/2 * sy + (nz-1)/2 * sz;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset<ndim-4>(i, size, stride);
            out_offset += offset;

            RegFieldStatic<C, three, BX, BY, BZ>::add_kernel_absolute(
                out + out_offset, sc, kernel);
        }
    });
}


template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void sub_kernel_absolute_3d(
    scalar_t * out,
    const offset_t * size,
    const offset_t * stride,
    const reduce_t absolute[C])
{
    offset_t numel = prod<ndim-4>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t sx  = stride[ndim-4];
    offset_t sy  = stride[ndim-3];
    offset_t sz  = stride[ndim-2];
    offset_t sc  = stride[ndim-1];

    // move kernel to the stack
    reduce_t kernel[C];
    for (offset_t c=0; c<C; ++c) kernel[c] = absolute[c];

    offset_t offset = (nx-1)/2 * sx + (ny-1)/2 * sy + (nz-1)/2 * sz;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset<ndim-4>(i, size, stride);
            out_offset += offset;

            RegFieldStatic<C, three, BX, BY, BZ>::sub_kernel_absolute(
                out + out_offset, sc, kernel);
        }
    });
}

// --- ABSOLUTE: diagonal ----------------------------------------------

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void diag_absolute_3d(
    scalar_t * out,
    const offset_t * size,
    const offset_t * stride,
    const reduce_t absolute[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t sx  = stride[ndim-4];
    offset_t sy  = stride[ndim-3];
    offset_t sz  = stride[ndim-2];
    offset_t sc  = stride[ndim-1];

    // move kernel to the stack
    reduce_t kernel[C];
    for (offset_t c=0; c<C; ++c) kernel[c] = absolute[c];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset_3d<ndim-1>(i, size, stride, x, y, z);
            out_offset += x * sx + y * sy + z * sz;

            RegFieldStatic<C, three, BX, BY, BZ>::diag_absolute(
                out + out_offset, sc, kernel);
        }
    });
}


template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void add_diag_absolute_3d(
    scalar_t * out,
    const offset_t * size,
    const offset_t * stride,
    const reduce_t absolute[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t nc  = size[ndim-1];
    offset_t sx  = stride[ndim-4];
    offset_t sy  = stride[ndim-3];
    offset_t sz  = stride[ndim-2];
    offset_t sc  = stride[ndim-1];

    // move kernel to the stack
    reduce_t kernel[C];
    for (offset_t c=0; c<C; ++c) kernel[c] = absolute[c];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset_3d<ndim-1>(i, size, stride, x, y, z);
            out_offset += x * sx + y * sy + z * sz;

            RegFieldStatic<C, three, BX, BY, BZ>::add_diag_absolute(
                out + out_offset, sc, kernel);
        }
    });
}


template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void sub_diag_absolute_3d(
    scalar_t * out,
    const offset_t * size,
    const offset_t * stride,
    const reduce_t absolute[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t sx  = stride[ndim-4];
    offset_t sy  = stride[ndim-3];
    offset_t sz  = stride[ndim-2];
    offset_t sc  = stride[ndim-1];

    // move kernel to the stack
    reduce_t kernel[C];
    for (offset_t c=0; c<C; ++c) kernel[c] = absolute[c];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset_3d<ndim-1>(i, size, stride, x, y, z);
            out_offset += x * sx + y * sy + z * sz;

            RegFieldStatic<C, three, BX, BY, BZ>::sub_diag_absolute(
                out + out_offset, sc, kernel);
        }
    });
}


//======================================================================
//                              MEMBRANE
//======================================================================

// --- MEMBRANE: vel2mom -----------------------------------------------

template <int C, int ndim,
          bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void vel2mom_membrane_3d(
    scalar_t * out, const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    reduce_t vx, reduce_t vy, reduce_t vz,
    const reduce_t absolute[C], const reduce_t membrane[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    constexpr int K = RegFieldStatic<C, three, BX, BY, BZ>::kernelsize_membrane;
    reduce_t kernel[K*C];
    RegFieldStatic<C, three, BX, BY, BZ>::make_kernel_membrane(kernel, absolute, membrane, vx, vy, vz);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t inp_offset = index2offset_3d<ndim-1>(i, size, stride_inp, x, y, z);
            offset_t out_offset = index2offset<ndim-1>(i, size, stride_out);
            inp_offset += x * isx + y * isy + z * isz;

            RegFieldStatic<C, three, BX, BY, BZ>::vel2mom_membrane(
                out + out_offset, inp + inp_offset,
                x, nx, isx, y, ny, isy, z, nz, isz, osc, isc, kernel);
        }
    });
}

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void add_vel2mom_membrane_3d(
    scalar_t * out, const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    reduce_t vx, reduce_t vy, reduce_t vz,
    const reduce_t absolute[C], const reduce_t membrane[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    constexpr int K = RegFieldStatic<C, three, BX, BY, BZ>::kernelsize_membrane;
    reduce_t kernel[K*C];
    RegFieldStatic<C, three, BX, BY, BZ>::make_kernel_membrane(kernel, absolute, membrane, vx, vy, vz);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t inp_offset = index2offset_3d<ndim-1>(i, size, stride_inp, x, y, z);
            offset_t out_offset = index2offset<ndim-1>(i, size, stride_out);
            inp_offset += x * isx + y * isy + z * isz;

            RegFieldStatic<C, three, BX, BY, BZ>::add_vel2mom_membrane(
                out + out_offset, inp + inp_offset,
                x, nx, isx, y, ny, isy, z, nz, isz, osc, isc, kernel);
        }
    });
}

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void sub_vel2mom_membrane_3d(
    scalar_t * out, const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    reduce_t vx, reduce_t vy, reduce_t vz,
    const reduce_t absolute[C], const reduce_t membrane[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    constexpr int K = RegFieldStatic<C, three, BX, BY, BZ>::kernelsize_membrane;
    reduce_t kernel[K*C];
    RegFieldStatic<C, three, BX, BY, BZ>::make_kernel_membrane(kernel, absolute, membrane, vx, vy, vz);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t inp_offset = index2offset_3d<ndim-1>(i, size, stride_inp, x, y, z);
            offset_t out_offset = index2offset<ndim-1>(i, size, stride_out);
            inp_offset += x * isx + y * isy + z * isz;

            RegFieldStatic<C, three, BX, BY, BZ>::sub_vel2mom_membrane(
                out + out_offset, inp + inp_offset,
                x, nx, isx, y, ny, isy, z, nz, isz, osc, isc, kernel);
        }
    });
}

// --- MEMBRANE: kernel ------------------------------------------------

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void kernel_membrane_3d(
    scalar_t * out,
    const offset_t * size,
    const offset_t * stride,
    reduce_t vx, reduce_t vy, reduce_t vz,
    const reduce_t absolute[C], const reduce_t membrane[C])
{
    offset_t numel = prod<ndim-4>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t sx  = stride[ndim-4];
    offset_t sy  = stride[ndim-3];
    offset_t sz  = stride[ndim-2];
    offset_t sc  = stride[ndim-1];

    constexpr int K = RegFieldStatic<C, three, BX, BY, BZ>::kernelsize_membrane;
    reduce_t kernel[K*C];
    RegFieldStatic<C, three, BX, BY, BZ>::make_fullkernel_membrane(
        kernel, absolute, membrane, vx, vy, vz);

    offset_t offset = (nx-1)/2 * sx + (ny-1)/2 * sy + (nz-1)/2 * sz;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset<ndim-4>(i, size, stride);
            out_offset += offset;

            RegFieldStatic<C, three, BX, BY, BZ>::kernel_membrane(
                out + out_offset, sc, sx, sy, sz, kernel);
        }
    });
}

// --- MEMBRANE: diagonal ----------------------------------------------

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void diag_membrane_3d(
    scalar_t * out,
    const offset_t * size,
    const offset_t * stride,
    reduce_t vx, reduce_t vy, reduce_t vz,
    const reduce_t absolute[C], const reduce_t membrane[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t sx  = stride[ndim-4];
    offset_t sy  = stride[ndim-3];
    offset_t sz  = stride[ndim-2];
    offset_t sc  = stride[ndim-1];

    constexpr int K = RegFieldStatic<C, three, BX, BY, BZ>::kernelsize_membrane;
    reduce_t kernel[K*C];
    RegFieldStatic<C, three, BX, BY, BZ>::make_fullkernel_membrane(
        kernel, absolute, membrane, vx, vy, vz);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset_3d<ndim-1>(i, size, stride, x, y, z);
            out_offset += x * sx + y * sy + z * sz;

            RegFieldStatic<C, three, BX, BY, BZ>::diag_membrane(
                out + out_offset, sc, x, nx, y, ny, z, nz, kernel);
        }
    });
}

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void add_diag_membrane_3d(
    scalar_t * out,
    const offset_t * size,
    const offset_t * stride,
    reduce_t vx, reduce_t vy, reduce_t vz,
    const reduce_t absolute[C], const reduce_t membrane[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t sx  = stride[ndim-4];
    offset_t sy  = stride[ndim-3];
    offset_t sz  = stride[ndim-2];
    offset_t sc  = stride[ndim-1];

    constexpr int K = RegFieldStatic<C, three, BX, BY, BZ>::kernelsize_membrane;
    reduce_t kernel[K*C];
    RegFieldStatic<C, three, BX, BY, BZ>::make_fullkernel_membrane(
        kernel, absolute, membrane, vx, vy, vz);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset_3d<ndim-1>(i, size, stride, x, y, z);
            out_offset += x * sx + y * sy + z * sz;

            RegFieldStatic<C, three, BX, BY, BZ>::add_diag_membrane(
                out + out_offset, sc, x, nx, y, ny, z, nz, kernel);
        }
    });
}

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void sub_diag_membrane_3d(
    scalar_t * out,
    const offset_t * size,
    const offset_t * stride,
    reduce_t vx, reduce_t vy, reduce_t vz,
    const reduce_t absolute[C], const reduce_t membrane[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t sx  = stride[ndim-4];
    offset_t sy  = stride[ndim-3];
    offset_t sz  = stride[ndim-2];
    offset_t sc  = stride[ndim-1];

    constexpr int K = RegFieldStatic<C, three, BX, BY, BZ>::kernelsize_membrane;
    reduce_t kernel[K*C];
    RegFieldStatic<C, three, BX, BY, BZ>::make_fullkernel_membrane(
        kernel, absolute, membrane, vx, vy, vz);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset_3d<ndim-1>(i, size, stride, x, y, z);
            out_offset += x * sx + y * sy + z * sz;

            RegFieldStatic<C, three, BX, BY, BZ>::sub_diag_membrane(
                out + out_offset, sc, x, nx, y, ny, z, nz, kernel);
        }
    });
}

//======================================================================
//                              BENDING
//======================================================================

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void vel2mom_bending_3d(
    scalar_t * out, const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    reduce_t vx, reduce_t vy, reduce_t vz,
    const reduce_t absolute[C], const reduce_t membrane[C], const reduce_t bending[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    constexpr int K = RegFieldStatic<C, three, BX, BY, BZ>::kernelsize_bending;
    reduce_t kernel[K*C];
    RegFieldStatic<C, three, BX, BY, BZ>::make_kernel_bending(
        kernel, absolute, membrane, bending, vx, vy, vz);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t inp_offset = index2offset_3d<ndim-1>(i, size, stride_inp, x, y, z);
            offset_t out_offset = index2offset<ndim-1>(i, size, stride_out);
            inp_offset += x * isx + y * isy + z * isz;

            RegFieldStatic<C, three, BX, BY, BZ>::vel2mom_bending(
                out + out_offset, inp + inp_offset,
                x, nx, isx, y, ny, isy, z, nz, isz, osc, isc, kernel);
        }
    });
}

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void kernel_bending_3d(
    scalar_t * out,
    const offset_t * size,
    const offset_t * stride,
    reduce_t vx, reduce_t vy, reduce_t vz,
    const reduce_t absolute[C], const reduce_t membrane[C], const reduce_t bending[C])
{
    offset_t numel = prod<ndim-4>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t sx  = stride[ndim-4];
    offset_t sy  = stride[ndim-3];
    offset_t sz  = stride[ndim-2];
    offset_t sc  = stride[ndim-1];

    constexpr int K = RegFieldStatic<C, three, BX, BY, BZ>::kernelsize_bending;
    reduce_t kernel[K*C];
    RegFieldStatic<C, three, BX, BY, BZ>::make_fullkernel_bending(
        kernel, absolute, membrane, bending, vx, vy, vz);

    offset_t offset = (nx-1)/2 * sx + (ny-1)/2 * sy + (nz-1)/2 * sz;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t out_offset = index2offset<ndim-4>(i, size, stride);
            out_offset += offset;

            RegFieldStatic<C, three, BX, BY, BZ>::kernel_bending(
                out + out_offset, sc, sx, sy, sz, kernel);
        }
    });
}

//======================================================================
//                           MEMBRANE JRLS
//======================================================================

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void vel2mom_membrane_jrls_3d(
    scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    const offset_t * stride_wgt,
    reduce_t vx, reduce_t vy, reduce_t vz,
    const reduce_t absolute[C], const reduce_t membrane[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t wsx = stride_wgt[ndim-4];
    offset_t wsy = stride_wgt[ndim-3];
    offset_t wsz = stride_wgt[ndim-2];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    constexpr int K = RegFieldStatic<C, three, BX, BY, BZ>::kernelsize_membrane_jrls;
    reduce_t kernel[K*C];
    RegFieldStatic<C, three, BX, BY, BZ>::make_kernel_membrane_jrls(kernel, absolute, membrane, vx, vy, vz);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t inp_offset = index2offset_3d<ndim-1>(i, size, stride_inp, x, y, z);
            offset_t out_offset = index2offset<ndim-1>(i, size, stride_out);
            offset_t wgt_offset = index2offset<ndim-1>(i, size, stride_wgt);
            inp_offset += x * isx + y * isy + z * isz;

            RegFieldStatic<C, three, BX, BY, BZ>::vel2mom_membrane_jrls(
                out + out_offset, inp + inp_offset, wgt + wgt_offset,
                x, nx, isx, wsx, y, ny, isy, wsy, z, nz, isz, wsz,
                osc, isc, kernel);
        }
    });
}

//======================================================================
//                           MEMBRANE RLS
//======================================================================

template <int C, int ndim, bound::type BX, bound::type BY, bound::type BZ,
          typename reduce_t, typename scalar_t, typename offset_t>
void vel2mom_membrane_rls_3d(
    scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    const offset_t * stride_wgt,
    reduce_t vx, reduce_t vy, reduce_t vz,
    const reduce_t absolute[C], const reduce_t membrane[C])
{
    offset_t numel = prod<ndim-1>(size);  // no outer loop across channels

    offset_t nx  = size[ndim-4];
    offset_t ny  = size[ndim-3];
    offset_t nz  = size[ndim-2];
    offset_t wsx = stride_wgt[ndim-4];
    offset_t wsy = stride_wgt[ndim-3];
    offset_t wsz = stride_wgt[ndim-2];
    offset_t wsc = stride_wgt[ndim-1];
    offset_t isx = stride_inp[ndim-4];
    offset_t isy = stride_inp[ndim-3];
    offset_t isz = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    constexpr int K = RegFieldStatic<C, three, BX, BY, BZ>::kernelsize_membrane_jrls;
    reduce_t kernel[K*C];
    RegFieldStatic<C, three, BX, BY, BZ>::make_kernel_membrane_jrls(kernel, absolute, membrane, vx, vy, vz);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t x, y, z;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t inp_offset = index2offset_3d<ndim-1>(i, size, stride_inp, x, y, z);
            offset_t out_offset = index2offset<ndim-1>(i, size, stride_out);
            offset_t wgt_offset = index2offset<ndim-1>(i, size, stride_wgt);
            inp_offset += x * isx + y * isy + z * isz;

            RegFieldStatic<C, three, BX, BY, BZ>::vel2mom_membrane_rls(
                out + out_offset, inp + inp_offset, wgt + wgt_offset,
                x, nx, isx, wsx, y, ny, isy, wsy, z, nz, isz, wsz,
                osc, isc, wsc, kernel);
        }
    });
}


} // namespace stat
} // namespace reg_grid
} // namespace jf

#endif // JF_REGULARISERS_FIELD_S_3D_LOOP
