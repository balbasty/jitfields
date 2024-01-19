from torch import Tensor

"""
Boundary conditions
-------------------
0: 'Zero'
1: 'Replicate'
2: 'DCT1'
3: 'DCT2'
4: 'DST1'
5: 'DST2'
6: 'DFT'
7: 'NoCheck'

Extrapolation methods
---------------------
-1: Discard data outside of the edges of the corner voxels
 0: Discard data outside of the centers of the corner voxels
+1: No not discard any data. Use boundary conditions to extrapolate
"""


def pull(out: Tensor, inp: Tensor, grid: Tensor,
         order: list[int], bound: list[int], extrapolate: int) -> Tensor:
    """
    Pull/sample a volume.

    Parameters
    ----------
    out : (*batch, *spatial_out, channels) tensor
        Preallocated output placeholder.
    inp : (*batch, *spatial_in, channels) tensor
        Input image to sample.
    grid : (*batch, *spatial_out, ndim) tensor
        Volume of voxel coordinates into the input volume.
    order : (ndim,) list[0..7]
        Spline order along each spatial dimension.
    bound : (ndim,) list[0..1]
        Boundary condition along each spatial dimension.
    extrapolate : {-1, 0, 1}
        Extrapolation method.

    Returns
    -------
    out : (*batch, *spatial_out, channels) tensor
        Sampled volume.
    """
    ...


def push(out: Tensor, inp: Tensor, grid: Tensor,
         order: list[int], bound: list[int], extrapolate: int) -> Tensor:
    """
    Push/splat a volume.

    Parameters
    ----------
    out : (*batch, *spatial_out, channels) tensor
        Preallocated output placeholder.
    inp : (*batch, *spatial_in, channels) tensor
        Input image to splat.
    grid : (*batch, *spatial_in, ndim) tensor
        Volume of voxel coordinates into the output volume.
    order : (ndim,) list[int]
    bound : (ndim,) list[int]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out : (*batch, *spatial_out, channels) tensor
        Splatted image
    """
    ...


def count(out: Tensor, grid: Tensor,
          order: list[int], bound: list[int], extrapolate: int) -> Tensor:
    """
    Push/splat a volume of ones

    Parameters
    ----------
    out : (*batch, *spatial_out, 1) tensor
    grid : (*batch, *spatial_in, ndim) tensor
    order : (ndim,) list[0..7]
    bound : (ndim,) list[0..7]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out : (*batch, *spatial_out, 1) tensor
    """
    ...


def grad(out: Tensor, inp: Tensor, grid: Tensor,
         order: list[int], bound: list[int], extrapolate: int, abs: bool):
    """
    Pull/sample the spatial gradients of an image

    Parameters
    ----------
    out : (*batch, *spatial_out, channels, ndim) tensor
    inp : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[0..7]
    bound : (ndim,) list[0..7]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out : (*batch, *spatial_out, channels, ndim) tensor
    """
    ...


def hess(out: Tensor, inp: Tensor, grid: Tensor,
         order: list[int], bound: list[int], extrapolate: int, abs: bool):
    """
    Pull/sample the spatial Hessians of an image

    Parameters
    ----------
    out : (*batch, *spatial_out, channels, ndim*(ndim+1)//2) tensor
    inp : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[0..7]
    bound : (ndim,) list[0..7]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out : (*batch, *spatial_out, channels, ndim*(ndim+1)//2) tensor
    """
    ...


def pull_backward(out_grad_inp: Tensor, out_grad_grid: Tensor,
                  inp_grad: Tensor, inp: Tensor, grid: Tensor,
                  order: list[int], bound: list[int], extrapolate: int,
                  abs: bool) -> tuple[Tensor, Tensor]:
    """
    Backward pass of `pull`

    Parameters
    ----------
    out_grad_inp : (*batch, *spatial_in, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    inp_grad : (*batch, *spatial_in, channels) tensor
    inp : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[0..7]
    bound : (ndim,) list[0..7]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out_grad_inp : (*batch, *spatial_in, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ...


def push_backward(out_grad_inp: Tensor, out_grad_grid: Tensor,
                  inp_grad: Tensor, inp: Tensor, grid: Tensor,
                  order: list[int], bound: list[int], extrapolate: int,
                  abs: bool) -> tuple[Tensor, Tensor]:
    """
    Backward pass of `push`

    Parameters
    ----------
    out_grad_inp : (*batch, *spatial_out, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    inp_grad : (*batch, *spatial_in, channels) tensor
    inp : (*batch, *spatial_out, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[0..7]
    bound : (ndim,) list[0..7]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out_grad_inp : (*batch, *spatial_out, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ...


def count_backward(out_grad_grid: Tensor, inp_grad: Tensor, grid: Tensor,
                   order: list[int], bound: list[int], extrapolate: int,
                   abs: bool) -> Tensor:
    """
    Backward pass of `count`

    Parameters
    ----------
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    inp_grad : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[0..7]
    bound : (ndim,) list[0..7]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ...


def grad_backward(out_grad_inp: Tensor, out_grad_grid: Tensor,
                  inp_grad: Tensor, inp: Tensor, grid: Tensor,
                  order: list[int], bound: list[int], extrapolate: int,
                  abs: bool) -> tuple[Tensor, Tensor]:
    """
    Backward pass of `grad`

    Parameters
    ----------
    out_grad_inp : (*batch, *spatial_in, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    inp_grad : (*batch, *spatial_in, channels, ndim) tensor
    inp : (*batch, *spatial_in, channels) tensor
    grid : (*batch, *spatial_out, ndim) tensor
    order : (ndim,) list[0..7]
    bound : (ndim,) list[0..7]
    extrapolate : {-1, 0, 1}

    Returns
    -------
    out_grad_inp : (*batch, *spatial_in, channels) tensor
    out_grad_grid : (*batch, *spatial_out, ndim) tensor
    """
    ...
