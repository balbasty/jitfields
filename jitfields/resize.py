try:
    from .cuda import resize as cuda_resize
except ImportError:
    cuda_resize = None
try:
    from .numba import resize as cpu_resize
except ImportError:
    cpu_resize = None
from .splinc import spline_coeff_nd
from .utils import ensure_list


def resize(x, factor=None, shape=None, ndim=None,
           anchor='e', order=2, bound='dct2', prefilter=True):
    """Resize a tensor using spline interpolation

    Parameters
    ----------
    x : (..., *inshape) tensor
        Input  tensor
    factor : [sequence of] float, optional
        Factor by which to resize the tensor (> 1 == bigger)
        One of factor or shape must be provided.
    shape : [sequence of] float, optional
        Shape of output tensor.
        One of factor or shape must be provided.
    ndim : int, optional
        Number if spatial dimensions.
        If not provided, try to guess from factor or shape.
        If guess fails, assume ndim = x.dim().
    anchor : {'edge', 'center'} or None
        What feature should be aligned across the input and output tensors.
        If 'edge' or 'center', the effective scaling factor may slightly
        differ from the requested scaling factor.
        If None, the center of the (0, 0) voxel is aligned, and the
        requested factor is exactly applied.
    order : [sequence of] {0..7}, default=2
        Interpolation order.
    bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
        How to deal with out-of-bound values.
    prefilter : bool, default=True
        Whether to first compute interpolating coefficients.
        Must be true for proper interpolation, otherwise this
        function merely performs a non-interpolating "prolongation".

    Returns
    -------
    x : (..., *shape) tensor
        Resized tensor

    """
    if x.is_cuda:
        _resize = cuda_resize.resize
    else:
        _resize = cpu_resize.resize

    if not ndim:
        if shape and hasattr(shape, '__len__'):
            ndim = len(shape)
        elif factor and hasattr(factor, '__len__'):
            ndim = len(shape)
        else:
            ndim = x.dim()
    if shape:
        shape = ensure_list(shape, ndim)
    elif factor:
        factor = ensure_list(factor, ndim)
    else:
        raise ValueError('At least one of shape or factor must be provided')
    if not shape:
        if out is not None:
            shape = out.shape[-ndim:]
        else:
            shape = [pymath.ceil(s*f) for s, f in zip(x.shape[-ndim:], factor)]

    if prefilter:
        spline_coeff_nd(x, order, bound, ndim)
    return _resize(x, factor, shape, ndim, anchor, order, bound)
