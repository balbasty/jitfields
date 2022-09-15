#ifndef JF_BATCH
#define JF_BATCH

#include <cuda_fp16.h>

template <typename T>
inline __device__
T square(T a) { return a*a; }

template <typename T>
inline __device__
T min(T a, T b) { return (a < b ? a : b); }

template <typename T>
inline __device__
T remainder(T x, T d)
{
    return (x - (x / d) * d);
}

template <typename T, typename size_t>
inline __device__
T prod(const T * x, size_t size)
{
    T tmp = x[0];
    for (size_t d = 1; d < size; ++d)
        tmp *= x[d];
    return tmp;
}

template <typename offset_t>
__device__
offset_t index2offset(offset_t index, int ndim,
                      const offset_t * size, const offset_t * stride)
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

template <typename offset_t>
__device__
offset_t index2offset_nd(offset_t index, int ndim,
                         const offset_t * size,
                         const offset_t * stride,
                         offset_t[] & x, int n)
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

#endif // JF_BATCH
