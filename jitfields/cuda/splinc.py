from ..common.bounds import convert_bound, cnames as bound_names
from ..common.spline import convert_order
from ..common.utils import cinfo
from ..utils import ensure_list, prod
from .utils import (culaunch, get_offset_type, load_code, to_cupy)
from ..common.splinc import get_poles
import cupy as cp


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
    shape, stride = cinfo(cu, dtype=offset_t, backend=cp)

    poles = cp.asarray(get_poles(order), dtype='double')
    npoles = cp.int(len(poles))

    # dispatch
    nvox = prod(cu.shape[:-1])
    kernel = get_kernel(bound, scalar_t, offset_t)
    culaunch(kernel, nvox, (cu, cp.int(cu.ndim), shape, stride, poles, npoles))

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
