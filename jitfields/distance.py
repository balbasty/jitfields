try:
    from .cuda import distance as cuda_dist
except ImportError:
    cuda_dist = None
try:
    # from .numba import distance as cpu_dist
    from .cpp import distance as cpu_dist
except ImportError:
    cpu_dist = None
import torch
from .utils import make_vector


def euclidean_distance_transform(x, dim=None, vx=1, dtype=None):
    """Compute the Euclidean distance transform of a binary image

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor
    dim : int, default=`x.dim()`
        Number of spatial dimensions
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    d : (..., *spatial) tensor
        Distance map

    References
    ----------
    ..[1] "Distance Transforms of Sampled Functions"
          Pedro F. Felzenszwalb & Daniel P. Huttenlocher
          Theory of Computing (2012)
          https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    if x.is_cuda:
        edt_1d_ = cuda_dist.edt_1d_
        l1dt_1d_ = cuda_dist.l1dt_1d_
    else:
        edt_1d_ = cpu_dist.edt_1d_
        l1dt_1d_ = cpu_dist.l1dt_1d_
    dtype = dtype or x.dtype
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    x = x.to(dtype, copy=True)
    x.masked_fill_(x > 0, float('inf'))
    dim = dim or x.dim()
    vx = make_vector(vx, dim, dtype=torch.float).tolist()
    x = l1dt_1d_(x, -dim, vx[0])
    if dim > 1:
        x.square_()
        for d, w in zip(range(1, dim), vx[1:]):
            x = edt_1d_(x, d-dim, w)
        x.sqrt_()
    return x


def l1_distance_transform(x, dim=None, vx=1, dtype=None):
    """Compute the L1 distance transform of a binary image

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor
    dim : int, default=`x.dim()`
        Number of spatial dimensions
    vx : [sequence of] float, default=1
        Voxel size
    dtype : torch.dtype
        Datatype of the distance map.
        By default, use x.dtype if it is a floating point type,
        otherwise use the default floating point type.

    Returns
    -------
    d : (..., *spatial) tensor
        Distance map

    References
    ----------
    ..[1] "Distance Transforms of Sampled Functions"
          Pedro F. Felzenszwalb & Daniel P. Huttenlocher
          Theory of Computing (2012)
          https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    if x.is_cuda:
        l1dt_1d_ = cuda_dist.l1dt_1d_
    else:
        l1dt_1d_ = cpu_dist.l1dt_1d_
    dtype = dtype or x.dtype
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    x = x.to(dtype, copy=True)
    x.masked_fill_(x > 0, float('inf'))
    dim = dim or x.dim()
    vx = make_vector(vx, dim, dtype=torch.float).tolist()
    for d, w in enumerate(vx):
        x = l1dt_1d_(x, d-dim, w)
    return x


def signed_distance_transform(x, dim=None, vx=1, dtype=None):
    """Compute the Euclidean distance transform of a binary image

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor
    dim : int, default=`x.dim()`
        Number of spatial dimensions
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    d : (..., *spatial) tensor
        Distance map

    References
    ----------
    ..[1] "Distance Transforms of Sampled Functions"
          Pedro F. Felzenszwalb & Daniel P. Huttenlocher
          Theory of Computing (2012)
          https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    x = x > 0
    d = euclidean_distance_transform(x, dim, vx, dtype)
    d -= euclidean_distance_transform(x.logical_not_(), dim, vx, dtype)
    return d
