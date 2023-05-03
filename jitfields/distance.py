__all__ = [
    'euclidean_distance_transform',
    'l1_distance_transform',
    'signed_distance_transform',
]

import torch
from torch import Tensor as tensor
from typing import Optional
from .utils import make_vector, try_import
from .utils import OneOrSeveral
cuda_dist = try_import('jitfields.bindings.cuda', 'distance')
cpu_dist = try_import('jitfields.bindings.cpp', 'distance')


def euclidean_distance_transform(
    x: tensor,
    ndim: Optional[int] = None,
    vx: OneOrSeveral[float] = 1,
    dtype: Optional[torch.dtype] = None,
) -> tensor:
    """Compute the Euclidean distance transform of a binary image

    Parameters
    ----------
    x : `(..., *spatial) tensor`
        Input tensor, with shape `(..., *spatial)`.
    ndim : `int`, default=`x.ndim`
        Number of spatial dimensions. Default: all.
    vx : `[sequence of] float`, default=1
        Voxel size.
    dtype : `torch.dtype`, optional
        Ouptut data type. Default is same as `x` if it has a floating
        point data type, else `torch.get_default_dtype()`.

    Returns
    -------
    d : `(..., *spatial) tensor`
        Distance map, with shape `(..., *spatial)`.

    References
    ----------
    1. Felzenszwalb, P.F. and Huttenlocher, D.P., 2012.
    [*Distance transforms of sampled functions.*](https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf)
    _Theory of computing_, 8(1), pp.415-428.

            @article{felzenszwalb2012distance,
              title={Distance transforms of sampled functions},
              author={Felzenszwalb, Pedro F and Huttenlocher, Daniel P},
              journal={Theory of computing},
              volume={8},
              number={1},
              pages={415--428},
              year={2012},
              publisher={Theory of Computing Exchange},
              url={https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf}
            }
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
    ndim = ndim or x.ndim
    vx = make_vector(vx, ndim, dtype=torch.float).tolist()
    x = l1dt_1d_(x, -ndim, vx[0])
    if ndim > 1:
        x.square_()
        for d, w in zip(range(1, ndim), vx[1:]):
            x = edt_1d_(x, d-ndim, w)
        x.sqrt_()
    return x


def l1_distance_transform(
    x: tensor,
    ndim: Optional[int] = None,
    vx: OneOrSeveral[float] = 1,
    dtype: Optional[torch.dtype] = None,
) -> tensor:
    """Compute the L1 distance transform of a binary image

    Parameters
    ----------
    x : `(..., *spatial) tensor`
        Input tensor, with shape `(..., *spatial)`.
    ndim : `int`, default=`x.ndim`
        Number of spatial dimensions. Default: all.
    vx : `[sequence of] float`, default=1
        Voxel size.
    dtype : `torch.dtype`, optional
        Ouptut data type. Default is same as `x` if it has a floating
        point data type, else `torch.get_default_dtype()`.

    Returns
    -------
    d : `(..., *spatial) tensor`
        Distance map, with shape `(..., *spatial)`.

    References
    ----------
    1. Felzenszwalb, P.F. and Huttenlocher, D.P., 2012.
    [*Distance transforms of sampled functions.*](https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf)
    _Theory of computing_, 8(1), pp.415-428.

            @article{felzenszwalb2012distance,
              title={Distance transforms of sampled functions},
              author={Felzenszwalb, Pedro F and Huttenlocher, Daniel P},
              journal={Theory of computing},
              volume={8},
              number={1},
              pages={415--428},
              year={2012},
              publisher={Theory of Computing Exchange},
              url={https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf}
            }
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
    ndim = ndim or x.ndim
    vx = make_vector(vx, ndim, dtype=torch.float).tolist()
    for d, w in enumerate(vx):
        x = l1dt_1d_(x, d-ndim, w)
    return x


def signed_distance_transform(x, ndim=None, vx=1, dtype=None):
    """Compute the Euclidean distance transform of a binary image

    Parameters
    ----------
    x : `(..., *spatial) tensor`
        Input tensor, with shape `(..., *spatial)`.
    ndim : `int`, default=`x.ndim`
        Number of spatial dimensions. Default: all.
    vx : `[sequence of] float`, default=1
        Voxel size.
    dtype : `torch.dtype`, optional
        Ouptut data type. Default is same as `x` if it has a floating
        point data type, else `torch.get_default_dtype()`.

    Returns
    -------
    d : `(..., *spatial) tensor`
        Distance map, with shape `(..., *spatial)`.

    References
    ----------
    1. Felzenszwalb, P.F. and Huttenlocher, D.P., 2012.
    [*Distance transforms of sampled functions.*](https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf)
    _Theory of computing_, 8(1), pp.415-428.

            @article{felzenszwalb2012distance,
              title={Distance transforms of sampled functions},
              author={Felzenszwalb, Pedro F and Huttenlocher, Daniel P},
              journal={Theory of computing},
              volume={8},
              number={1},
              pages={415--428},
              year={2012},
              publisher={Theory of Computing Exchange},
              url={https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf}
            }
    """
    x = x > 0
    d = euclidean_distance_transform(x, ndim, vx, dtype)
    d -= euclidean_distance_transform(x.logical_not_(), ndim, vx, dtype)
    return d
