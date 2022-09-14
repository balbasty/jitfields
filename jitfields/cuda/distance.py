from .utils import get_cuda_num_threads, get_cuda_blocks
from torch.utils.dlpack import to_dlpack
import cupy as cp
import numpy as np
import math as pymath
import torch


common_code = r"""
#include <cuda_fp16.h>
using namespace std;

template <typename T> 
__forceinline__ __device__ 
T square(T a) { return a*a; } 

template <typename T> 
__forceinline__ __device__ 
T min(T a, T b) { return (a < b ? a : b); } 

template <typename T> 
__forceinline__ __device__ 
T remainder(T x, T d)
{ 
    return (x - (x / d) * d); 
} 

template <typename T, typename size_t> 
__forceinline__ __device__ 
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
"""

l1dt_kernel_code = common_code + r"""
template <typename offset_t, typename scalar_t>
__device__ 
void algo(scalar_t * f, offset_t size, offset_t stride, scalar_t w) 
{
  if (size == 1) return;
  
  scalar_t tmp = *f;
  f += stride;
  for (offset_t i = 1; i < size; ++i, f += stride) {
     tmp = min(tmp + w, *f);
     *f = tmp;
  }
  f -= stride;
  for (offset_t i = 1; i < size; ++i, f -= stride) {
     tmp = min(tmp + w, *f);
     *f = tmp;
  }
}


template <typename scalar_t, typename offset_t>
__global__ void kernel(scalar_t * f, scalar_t w, int ndim, 
                       const offset_t * size, const offset_t *  stride) 
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size, ndim-1);

    for (offset_t i=index; index < nthreads; 
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t batch_offset = index2offset(i, ndim-1, size, stride);
        algo(f + batch_offset, size[ndim-1], stride[ndim-1], w);
    }
}
"""

l1dt_templates = (
    'kernel<half,int>',
    'kernel<half,long>',
    'kernel<float,int>',
    'kernel<float,long>',
    'kernel<double,int>',
    'kernel<double,long>',
)

l1dt_module = cp.RawModule(code=l1dt_kernel_code,
                           options=('--std=c++11',),
                           name_expressions=l1dt_templates)
l1dt_kernels = {
    (cp.dtype('float16'), cp.int32): l1dt_module.get_function('kernel<half,int>'),
    (cp.dtype('float16'), cp.int64): l1dt_module.get_function('kernel<half,long>'),
    (cp.dtype('float32'), cp.int32): l1dt_module.get_function('kernel<float,int>'),
    (cp.dtype('float32'), cp.int64): l1dt_module.get_function('kernel<float,long>'),
    (cp.dtype('float64'), cp.int32): l1dt_module.get_function('kernel<double,int>'),
    (cp.dtype('float64'), cp.int64): l1dt_module.get_function('kernel<double,long>'),
}


def l1dt_1d_(f, dim=-1, w=1):
    """in-place one-dimensional L1 distance"""
    f = f.movedim(dim, -1)
    n = pymath.prod(f.shape[:-1])
    cuf = cp.from_dlpack(to_dlpack(f))
    shape = cp.asarray(cuf.shape)
    stride = cp.asarray([s // cuf.dtype.itemsize for s in cuf.strides])
    if n <= cp.iinfo('int32').max:
        kernel = l1dt_kernels[(cuf.dtype, cp.int32)]
        shape = shape.astype(cp.int32)
        stride = stride.astype(cp.int32)
    else:
        kernel = l1dt_kernels[(cuf.dtype, cp.int64)]
        shape = shape.astype(cp.int64)
        stride = stride.astype(cp.int64)
    kernel((get_cuda_blocks(n),), (get_cuda_num_threads(),),
           (cuf, cuf.dtype.type(w), cp.int(cuf.ndim), shape, stride))
    f = f.movedim(-1, dim)
    return f


def l1dt_1d(f, dim=-1, w=1):
    """out-of-place one-dimensional L1 distance"""
    dtype = f.dtype
    if not f.dtype.is_floating_point:
        f = torch.get_default_dtype()
    f = f.to(dtype, copy=True)
    return l1dt_1d_(f, dim, w)


edt_kernel_code = common_code + r"""

template <typename out_t, typename in_t>
__device__ __forceinline__
out_t mycast(in_t x) { return static_cast<out_t>(static_cast<float>(x)); }

template <typename offset_t, typename scalar_t>
__device__ 
scalar_t intersection(scalar_t * f, offset_t * v, scalar_t * z, scalar_t w2,
                      offset_t k, offset_t q,
                      offset_t size, offset_t stride, offset_t stride_buf) 
{
    offset_t vk = v[k * stride_buf];
    scalar_t fvk = f[vk * stride];
    scalar_t fq = f[q * stride];
    offset_t a = q - vk, b = q + vk;
    scalar_t s = fq - mycast<scalar_t>(fvk);
    s += w2 * mycast<scalar_t>(a * b);
    s /= mycast<scalar_t>(2) * w2 * mycast<scalar_t>(a);
    return s;
}

template <typename offset_t, typename scalar_t>
__device__ 
void fillin(scalar_t * f, offset_t * v, scalar_t * z, scalar_t * d, scalar_t w2,
            offset_t size, offset_t stride, offset_t stride_buf) 
{
    offset_t k = 0;
    offset_t vk;
    for (offset_t q = 0; q < size; ++q) {
        scalar_t fq = mycast<scalar_t>(q);
        while ((k < size-1) && (z[(k+1) * stride_buf] < fq)) {
            ++k;
        }
        vk = v[k * stride_buf];
        d[q * stride_buf] = f[vk * stride] 
                          + w2 * mycast<scalar_t>(square(q - vk));
    }
    for (offset_t q = 0; q < size; ++q)
        f[q * stride] = d[q * stride_buf];
}

template <typename offset_t, typename scalar_t>
__device__ 
void algo(scalar_t * f, offset_t * v, scalar_t * z, scalar_t * d, scalar_t w2,
          offset_t size, offset_t stride, offset_t stride_buf) 
{
    if (size == 1) return;

    v[0] = 0;
    z[0] = -(1./0.);
    z[stride_buf] = 1./0.;
    scalar_t s;
    offset_t k = 0;
    for (offset_t q=1; q < size; ++q) {
        while (1) {
            s = intersection(f, v, z, w2, k, q, size, stride, stride_buf);
            if ((k == 0) || (s > z[k * stride_buf]))
                break;
            --k;
        }
        if (isnan(static_cast<float>(s)))
            s = -(1./0.);
    
        ++k; 
        v[k * stride_buf] = q;
        z[k * stride_buf] = s;
        z[(k+1) * stride_buf] = 1./0.; 
    }
    fillin(f, v, z, d, w2, size, stride, stride_buf);
}


template <typename scalar_t, typename offset_t>
__global__ void kernel(scalar_t * f, char * buf, scalar_t w, int ndim, 
                       const offset_t * size, const offset_t *  stride) 
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size, ndim-1);
    
    offset_t n = size[ndim-1];
    offset_t stride_buf = blockDim.x * gridDim.x;
    offset_t * v = reinterpret_cast<offset_t *>(buf);
    scalar_t * z = reinterpret_cast<scalar_t *>(buf 
                 + stride_buf * n * sizeof(offset_t));
    scalar_t * d = reinterpret_cast<scalar_t *>(buf 
                 + stride_buf * n * (sizeof(offset_t) + sizeof(scalar_t)));
    
    w = w*w;

    for (offset_t i=index; index < nthreads; index += stride_buf, i=index)
    {
        offset_t batch_offset = index2offset(i, ndim-1, size, stride);
        algo(f + batch_offset, v + index, z + index, d + index, w,
             n, stride[ndim-1], stride_buf);
    }
}
"""

edt_templates = (
    'kernel<half,int>',
    'kernel<half,long>',
    'kernel<float,int>',
    'kernel<float,long>',
    'kernel<double,int>',
    'kernel<double,long>',
)

edt_module = cp.RawModule(code=edt_kernel_code,
                          options=('--std=c++11',),
                          name_expressions=edt_templates)
edt_kernels = {
    (cp.dtype('float16'), cp.int32): edt_module.get_function('kernel<half,int>'),
    (cp.dtype('float16'), cp.int64): edt_module.get_function('kernel<half,long>'),
    (cp.dtype('float32'), cp.int32): edt_module.get_function('kernel<float,int>'),
    (cp.dtype('float32'), cp.int64): edt_module.get_function('kernel<float,long>'),
    (cp.dtype('float64'), cp.int32): edt_module.get_function('kernel<double,int>'),
    (cp.dtype('float64'), cp.int64): edt_module.get_function('kernel<double,long>'),
}


def edt_1d_(f, dim=-1, w=1):
    """in-place one-dimensional Euclidean distance"""
    f = f.movedim(dim, -1)
    n = pymath.prod(f.shape[:-1])
    cuf = cp.from_dlpack(to_dlpack(f))
    shape = cp.asarray(cuf.shape)
    stride = cp.asarray([s // cuf.dtype.itemsize for s in cuf.strides])
    if n <= cp.iinfo('int32').max:
        kernel = edt_kernels[(cuf.dtype, cp.int32)]
        shape = shape.astype(cp.int32)
        stride = stride.astype(cp.int32)
    else:
        kernel = edt_kernels[(cuf.dtype, cp.int64)]
        shape = shape.astype(cp.int64)
        stride = stride.astype(cp.int64)
    nb_blocks, nb_threads = get_cuda_blocks(n), get_cuda_num_threads()
    buf = nb_blocks * nb_threads * f.shape[-1]
    buf *= 2 * cuf.dtype.itemsize + stride.dtype.itemsize
    buf = cp.empty([buf], dtype=cp.uint8)
    kernel((nb_blocks,), (nb_threads,),
           (cuf, buf, cuf.dtype.type(w), cp.int(cuf.ndim), shape, stride))
    f = f.movedim(-1, dim)
    return f


def edt_1d(f, dim=-1, w=1):
    """out-of-place one-dimensional Euclidean distance"""
    dtype = f.dtype
    if not f.dtype.is_floating_point:
        f = torch.get_default_dtype()
    f = f.to(dtype, copy=True)
    return edt_1d_(f, dim, w)