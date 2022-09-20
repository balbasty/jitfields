from .utils import (get_cuda_num_threads, get_cuda_blocks,
                    load_code, to_cupy)
import cupy as cp
import math as pymath
import torch

l1dt_templates = (
    'kernel<half,int>',
    'kernel<half,long>',
    'kernel<float,int>',
    'kernel<float,long>',
    'kernel<double,int>',
    'kernel<double,long>',
)

l1dt_module = cp.RawModule(code=load_code('distance_l1.cu'),
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
    cuf = to_cupy(f)
    shape = cp.asarray(cuf.shape)
    stride = cp.asarray([s // cuf.dtype.itemsize for s in cuf.strides])
    offset_t = cp.int32 if n <= cp.iinfo('int32').max else cp.int64
    kernel = l1dt_kernels[(cuf.dtype, offset_t)]
    shape = shape.astype(offset_t)
    stride = stride.astype(offset_t)
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

edt_templates = (
    'kernel<half,int>',
    'kernel<half,long>',
    'kernel<float,int>',
    'kernel<float,long>',
    'kernel<double,int>',
    'kernel<double,long>',
)

edt_module = cp.RawModule(code=load_code('distance_euclidean.cu'),
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
    cuf = to_cupy(f)
    shape = cp.asarray(cuf.shape)
    stride = cp.asarray([s // cuf.dtype.itemsize for s in cuf.strides])
    offset_t = cp.int32 if n <= cp.iinfo('int32').max else cp.int64
    kernel = edt_kernels[(cuf.dtype, offset_t)]
    shape = shape.astype(offset_t)
    stride = stride.astype(offset_t)
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
