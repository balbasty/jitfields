#ifndef JF_BATCH
#define JF_BATCH
#include "cuda_switch.h"
#include "utils.h"

/* Utilities to convert contiguous linear indices to
 * - sub-indices, and/or
 * - strided linear indices
 *
 * All functions assume that the input index is a linear index into a
 * contiguous volume of size `size` (with `ndim` dimensions), with a
 * Fortran layout. That is
 *      index = i + size[0] * (j + size[1] * (k + ...))
 *
 * The `stride` vector is then used to build the corresponding
 * strided linear index
 *      strided = i * stride[0] + j * stride[1] + k * stride[2] + ...
 *
 * Functions postfixed 1d/2d/3d only convert the leading "batch"
 * dimensions to a strided index, and return the remaining sub-indices
 * (x, y, z) in placeholders.
 *
 * Each function comes in two flavors:
 * - a dynamically sized version, where `ndim` is a function argument
 * - a statically sized version, where `ndim` is a template parameter
 */

namespace jf {

template <typename offset_t>
inline __device__
offset_t index2offset(offset_t index, int ndim,
                      const offset_t * size, const offset_t * stride)
{
    offset_t new_index = 0, new_index1;
    offset_t current_stride = 1, next_stride = 1;
    for (int i = 0; i < ndim; ++i) {
        new_index1 = index;
        next_stride = current_stride * size[i];
        new_index1 = remainder(index, next_stride);
        new_index1 = new_index1 / current_stride;
        current_stride = next_stride;
        new_index += new_index1 * stride[i];
    }
    return new_index;
}

template <int ndim, typename offset_t>
inline __device__
offset_t index2offset(offset_t index,
                      const offset_t * size,
                      const offset_t * stride)
{
    offset_t new_index = 0, new_index1;
    offset_t cur_stride = 1, nxt_stride = 1;
#   pragma unroll
    for (int i = 0; i < ndim; ++i) {
        new_index1 = index;
        nxt_stride = cur_stride * size[i];
        new_index1 = remainder(index, nxt_stride);
        new_index1 = new_index1 / cur_stride;
        cur_stride = nxt_stride;
        new_index += new_index1 * stride[i];
    }
    return new_index;
}

template <typename offset_t>
__device__
offset_t index2offset_1d(offset_t index, int ndim,
                         const offset_t * size,
                         const offset_t * stride,
                         offset_t & x)
{
    offset_t new_index = 0, new_index1 = 0;
    offset_t current_stride = 1, next_stride = 1;
    for (int i = 0; i < ndim; ++i) {
        new_index1 = index;
        if (i < ndim-1)  {
            next_stride = current_stride * size[i];
            new_index1 = remainder(index, next_stride);
        }
        new_index1 = new_index1 / current_stride;
        current_stride = next_stride;
        if (i < ndim-1)
            new_index += new_index1 * stride[i];
        else
            x = new_index1;
    }
    return new_index;
}

template <int ndim, typename offset_t>
__device__
offset_t index2offset_1d(offset_t index,
                         const offset_t * size,
                         const offset_t * stride,
                         offset_t & x)
{
    offset_t new_index = 0, new_index1 = 0;
    offset_t current_stride = 1, next_stride = 1;
#   pragma unroll
    for (int i = 0; i < ndim; ++i) {
        new_index1 = index;
        if (i < ndim-1)  {
            next_stride = current_stride * size[i];
            new_index1 = remainder(index, next_stride);
        }
        new_index1 = new_index1 / current_stride;
        current_stride = next_stride;
        if (i < ndim-1)
            new_index += new_index1 * stride[i];
        else
            x = new_index1;
    }
    return new_index;
}

template <typename offset_t>
__device__
offset_t index2offset_2d(offset_t index, int ndim,
                         const offset_t * size,
                         const offset_t * stride,
                         offset_t & x, offset_t & y)
{
    offset_t new_index = 0, new_index1;
    offset_t current_stride = 1, next_stride = 1;
    for (int i = 0; i < ndim; ++i) {
        new_index1 = index;
        if (i < ndim-1)  {
            next_stride = current_stride * size[i];
            new_index1 = remainder(index, next_stride);
        }
        new_index1 = new_index1 / current_stride;
        current_stride = next_stride;
        if (i < ndim-2)
            new_index += new_index1 * stride[i];
        else if (i == ndim-2)
            x = new_index1;
        else
            y = new_index1;
    }
    return new_index;
}

template <int ndim, typename offset_t>
inline __device__
offset_t index2offset_2d(offset_t index,
                         const offset_t * size,
                         const offset_t * stride,
                         offset_t & x, offset_t & y)
{
    offset_t new_index = 0, new_index1;
    offset_t current_stride = 1, next_stride = 1;
#   pragma unroll
    for (int i = 0; i < ndim; ++i) {
        new_index1 = index;
        if (i < ndim-1)  {
            next_stride = current_stride * size[i];
            new_index1 = remainder(index, next_stride);
        }
        new_index1 = new_index1 / current_stride;
        current_stride = next_stride;
        if (i < ndim-2)
            new_index += new_index1 * stride[i];
        else if (i == ndim-2)
            x = new_index1;
        else
            y = new_index1;
    }
    return new_index;
}

template <typename offset_t>
__device__
offset_t index2offset_3d(offset_t index, int ndim,
                         const offset_t * size,
                         const offset_t * stride,
                         offset_t & x, offset_t & y, offset_t & z)
{
    offset_t new_index = 0, new_index1;
    offset_t current_stride = 1, next_stride = 1;
    for (int i = 0; i < ndim; ++i) {
        new_index1 = index;
        if (i < ndim-1)  {
            next_stride = current_stride * size[i];
            new_index1 = remainder(index, next_stride);
        }
        new_index1 = new_index1 / current_stride;
        current_stride = next_stride;
        if (i < ndim-3)
            new_index += new_index1 * stride[i];
        else if (i == ndim-3)
            x = new_index1;
        else if (i == ndim-2)
            y = new_index1;
        else
            z = new_index1;
    }
    return new_index;
}

template <int ndim, typename offset_t>
inline __device__
offset_t index2offset_3d(offset_t index,
                         const offset_t * size,
                         const offset_t * stride,
                         offset_t & x, offset_t & y, offset_t & z)
{
    offset_t new_index = 0,  new_index1;
    offset_t cur_stride = 1, nxt_stride = 1;
#   pragma unroll
    for (int i = 0; i < ndim; ++i) {
        new_index1 = index;
        if (i < ndim-1)  {
            nxt_stride = cur_stride * size[i];
            new_index1 = remainder(index, nxt_stride);
        }
        new_index1 = new_index1 / cur_stride;
        cur_stride = nxt_stride;
        if (i < ndim-3)
            new_index += new_index1 * stride[i];
        else if (i == ndim-3)
            x = new_index1;
        else if (i == ndim-2)
            y = new_index1;
        else
            z = new_index1;
    }
    return new_index;
}

template <typename offset_t>
__device__
offset_t index2offset_nd(offset_t index, int ndim,
                         const offset_t * size,
                         const offset_t * stride,
                         offset_t * x, int n)
{
    offset_t new_index = 0, new_index1;
    offset_t current_stride = 1, next_stride = 1;
    for (int i = 0; i < ndim; ++i) {
        new_index1 = index;
        if (i < ndim-1)  {
            next_stride = current_stride * size[i];
            new_index1 = remainder(index, next_stride);
        }
        new_index1 = new_index1 / current_stride;
        current_stride = next_stride;
        if (i < ndim-n)
            new_index += new_index1 * stride[i];
        else
            x[ndim-n+i] = new_index1;
    }
    return new_index;
}


template <int n, int ndim, typename offset_t>
inline __device__
offset_t index2offset_nd(offset_t index,
                         const offset_t * size,
                         const offset_t * stride,
                         offset_t * x)
{
    offset_t new_index = 0, new_index1;
    offset_t current_stride = 1, next_stride = 1;
#   pragma unroll
    for (int i = 0; i < ndim; ++i) {
        new_index1 = index;
        if (i < ndim-1)  {
            next_stride = current_stride * size[i];
            new_index1 = remainder(index, next_stride);
        }
        new_index1 = new_index1 / current_stride;
        current_stride = next_stride;
        if (i < ndim-n)
            new_index += new_index1 * stride[i];
        else
            x[ndim-n+i] = new_index1;
    }
    return new_index;
}


template <int n, int ndim, typename offset_t>
inline __device__
offset_t index2offset_v2(
    offset_t index, const offset_t * size, const offset_t * stride,
    offset_t * x = nullptr)
{
    offset_t new_index = 0, new_index1;
    offset_t current_stride = 1, next_stride = 1;
#   pragma unroll
    for (int i = 0; i < ndim; ++i) {
        new_index1 = index;
        if (i < ndim-1)  {
            next_stride = current_stride * size[i];
            new_index1 = remainder(index, next_stride);
        }
        new_index1 = new_index1 / current_stride;
        current_stride = next_stride;
        if (i >= ndim-n)
            x[ndim-n+i] = new_index1;
        new_index += new_index1 * stride[i];
    }
    return new_index;
}


} // namespace jf

#endif // JF_BATCH
