from torch import Tensor
from numbers import Real


def l1dt_1d_(f: Tensor, dim: int = -1, w: Real = 1) -> Tensor:
    """
    Perform the one-dimensional L1 distance transform, in-place

    Parameters
    ----------
    f : (..., N) tensor
        The previous upper bound on the L1 distance.
        Initially, the background should contain zeros and the foreground
        should contain NaNs. The distance will be propagated into the NaNs.
    dim : int, default=-1
        Index of the dimension along which to perform the transform.
    w : real, default=1
        Voxel size along the transformed dimension.

    Returns
    -------
    f : (..., N) tensor
        Updated upper bound.
    """
    ...


def l1dt_1d(f: Tensor, dim: int = -1, w: Real = 1) -> Tensor:
    """Same as `l1dt_1d`, but carried out-of-place."""
    ...


def edt_1d_(f: Tensor, dim: int = -1, w: Real = 1) -> Tensor:
    """
    Perform the one-dimensional L2 distance transform, in-place

    Parameters
    ----------
    f : (..., N) tensor
        The previous upper bound on the squared L2 distance.
        Initially, the background should contain zeros and the foreground
        should contain NaNs. The distance will be propagated into the NaNs.
    dim : int, default=-1
        Index of the dimension along which to perform the transform.
    w : real, default=1
        Voxel size along the transformed dimension.

    Returns
    -------
    f : (..., N) tensor
        Updated upper bound.
    """
    ...
