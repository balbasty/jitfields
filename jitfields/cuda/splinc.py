from .bounds import convert_bound, cnames as bound_names
from .spline import convert_order
from ..utils import ensure_list, prod
from .utils import (get_cuda_blocks, get_cuda_num_threads, get_offset_type,
                    load_code, to_cupy)
import math as pymath
import cupy as cp


def get_poles(order):
    empty = []
    if order in (0, 1):
        return empty
    if order == 2:
        return [pymath.sqrt(8.) - 3.]
    if order == 3:
        return [pymath.sqrt(3.) - 2.]
    if order == 4:
        return [pymath.sqrt(664. - pymath.sqrt(438976.)) + pymath.sqrt(304.) - 19.,
                pymath.sqrt(664. + pymath.sqrt(438976.)) - pymath.sqrt(304.) - 19.]
    if order == 5:
        return [pymath.sqrt(67.5 - pymath.sqrt(4436.25)) + pymath.sqrt(26.25) - 6.5,
                pymath.sqrt(67.5 + pymath.sqrt(4436.25)) - pymath.sqrt(26.25) - 6.5]
    if order == 6:
        return [-0.488294589303044755130118038883789062112279161239377608394,
                -0.081679271076237512597937765737059080653379610398148178525368,
                -0.00141415180832581775108724397655859252786416905534669851652709]
    if order == 7:
        return [-0.5352804307964381655424037816816460718339231523426924148812,
                -0.122554615192326690515272264359357343605486549427295558490763,
                -0.0091486948096082769285930216516478534156925639545994482648003]
    raise NotImplementedError

# ===
# Build CUDA code
# ===

code = load_code('splinc.cu')

# ===
# Load module + dispatch
# ===
kernels = {}


def get_kernel(bound, scalar_t, offset_t):
    key = (bound, scalar_t, offset_t)
    if key not in kernels:
        template = f'kernel<'
        template += f'bound::type::{bound_names[bound]},'
        if scalar_t == cp.float32:
            template += 'float,'
        elif scalar_t == cp.float64:
            template += 'double,'
        elif scalar_t == cp.float16:
            template += 'half,'
        else:
            raise ValueError('Unknown scalar type', scalar_t)
        if offset_t == cp.int32:
            template += 'int>'
        elif offset_t == cp.int64:
            template += 'long>'
        else:
            raise ValueError('Unknown offset type', offset_t)
        module = cp.RawModule(code=code, options=('--std=c++14',),
                              name_expressions=(template,))
        kernel = module.get_function(template)
        kernels[key] = kernel
    else:
        kernel = kernels[key]

    return kernel


def spline_coeff_(inp, order, bound, dim=-1):
    order = convert_order.get(order, order)
    bound = convert_bound.get(bound, bound)
    if order in (0, 1):
        return inp

    cu = to_cupy(inp.movedim(dim, -1))
    offset_t = get_offset_type(cu.shape)
    scalar_t = cu.dtype.type

    shape = cp.asarray(cu.shape, dtype=offset_t)
    stride = [s // cp.dtype(cu.dtype).itemsize for s in cu.strides]
    stride = cp.asarray(stride, dtype=offset_t)

    poles = cp.asarray(get_poles(order), dtype='double')
    npoles = cp.int(len(poles))

    # dispatch
    nvox = prod(cu.shape[:-1])
    nblocks = get_cuda_blocks(nvox)
    nthreads = get_cuda_num_threads()
    nthreads = min(nthreads, int(pymath.ceil(nvox / nblocks)))
    kernel = get_kernel(bound, scalar_t, offset_t)
    kernel((nblocks,), (nthreads,),
           (cu, cp.int(cu.ndim), shape, stride, poles, npoles))

    return inp


def spline_coeff_nd_(inp, order, bound, ndim=None):
    if ndim is None:
        ndim = inp.dim()

    order = [convert_order.get(o, o) for o in ensure_list(order, ndim)]
    bound = [convert_bound.get(b, b) for b in ensure_list(bound, ndim)]
    if all([o in (0, 1) for o in order]):
        return inp

    for d, b, o in zip(range(ndim), reversed(bound), reversed(order)):
        spline_coeff_(inp, o, b, dim=-d-1)

    return inp