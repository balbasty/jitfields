import numba as nb


if hasattr(nb, 'get_thread_id'):
    get_thread_id = nb.get_thread_id
else:
    get_thread_id = nb.np.ufunc.parallel._get_thread_id


# https://stackoverflow.com/questions/61509903
@nb.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    """Returns a void pointer from a given memory address"""
    from numba.core import types, cgutils
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen


@nb.njit
def svector(ptr, length, stride=1, dtype=None):
    """Build a strided vector from a data pointer"""
    vec = nb.carray(address_as_void_pointer(ptr), (length, stride), dtype)
    return vec[:, 0]


@nb.njit
def vector(ptr, length, dtype=None):
    """Build a contiguous vector from a data pointer"""
    vec = nb.carray(address_as_void_pointer(ptr), length, dtype)
    return vec


@nb.njit
def remainder(x, d):
    return x - (x // d) * d


@nb.njit
def square(x):
    return x * x


@nb.njit
def prod(x):
    tmp = x[0]
    for i in range(1, len(x)):
        tmp *= x[i]
    return tmp


@nb.njit
def index2offset(index, shape, stride):
    ndim = len(shape)
    new_index = 0
    current_stride = next_stride = 1
    for i in range(ndim):
        new_index1 = index
        if i < ndim-1:
            next_stride = current_stride * shape[i]
            new_index1 = remainder(index, next_stride)
        new_index1 = new_index1 // current_stride
        current_stride = next_stride
        new_index += new_index1 * stride[i]
    return new_index
