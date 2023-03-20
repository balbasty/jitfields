from torch import Tensor


def flow_matvec(
    out: Tensor, inp: Tensor, bound: list[int], voxel_size: list[float],
    absolute: float, membrane: float, bending: float, shears: float, div: float,
    op: str = '=') -> Tensor: ...
"""
Forward matrix-vector product

Parameters
----------
out : (*batch, *spatial, ndim) tensor
inp : (*batch, *spatial, ndim) tensor
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : float
membrane : float
bending : float
shears : float
div : float
op : {'=', '+', '-'}

Returns
-------
out : (*batch, *spatial, ndim) tensor
"""

def flow_matvec_rls(
    out: Tensor, inp: Tensor, wgt: Tensor, bound: list[int], voxel_size: list[float],
    absolute: float, membrane: float, bending: float, shears: float, div: float,
    op: str = '=') -> Tensor: ...
"""
Forward matrix-vector product, with local weighting

Parameters
----------
out : (*batch, *spatial, ndim) tensor
inp : (*batch, *spatial, ndim) tensor
wgt : (*batch, *spatial) tensor
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : float
membrane : float
bending : float
shears : float
div : float
op : {'=', '+', '-'}

Returns
-------
out : (*batch, *spatial, ndim) tensor
"""

def flow_kernel(
    out: Tensor, bound: list[int], voxel_size: list[float],
    absolute: float, membrane: float, bending: float, shears: float, div: float,
    op: str = '=') -> Tensor: ...
"""
Equivalent convolution kernel

Parameters
----------
out : (*batch, *spatial, ndim) tensor
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : float
membrane : float
bending : float
shears : float
div : float
op : {'=', '+', '-'}

Returns
-------
out : (*batch, *spatial, ndim) tensor
"""

def flow_diag(
    out: Tensor, bound: list[int], voxel_size: list[float],
    absolute: float, membrane: float, bending: float, shears: float, div: float,
    op: str = '=') -> Tensor: ...
"""
Diagonal of the forward matrix

Parameters
----------
out : (*batch, *spatial, ndim) tensor
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : float
membrane : float
bending : float
shears : float
div : float
op : {'=', '+', '-'}

Returns
-------
out : (*batch, *spatial, ndim) tensor
"""

def flow_diag_rls(
    out: Tensor, wgt: Tensor, bound: list[int], voxel_size: list[float],
    absolute: float, membrane: float, bending: float, shears: float, div: float,
    op: str = '=') -> Tensor: ...
"""
Diagonal of the forward matrix, with local weighting

Parameters
----------
out : (*batch, *spatial, ndim) tensor
wgt : (*batch, *spatial) tensor
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : float
membrane : float
bending : float
shears : float
div : float
op : {'=', '+', '-'}

Returns
-------
out : (*batch, *spatial, ndim) tensor
"""

def flow_relax_(
    sol: Tensor, hes: Tensor, grd: Tensor, niter: int,
    bound: list[int], voxel_size: list[float],
    absolute: float, membrane: float, bending: float, shears: float, div: float,
    ) -> Tensor: ...
"""
Perform relaxation iterations

Parameters
----------
sol : (*batch, *spatial, ndim) tensor
hes : (*batch, *spatial, (ndim*(ndim+1))//2) tensor
grd : (*batch, *spatial, ndim) tensor
niter : int
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : float
membrane : float
bending : float
shears : float
div : float

Returns
-------
sol : (*batch, *spatial, ndim) tensor
"""

def flow_relax_rls_(
    sol: Tensor, hes: Tensor, grd: Tensor, wgt: Tensor, niter: int,
    bound: list[int], voxel_size: list[float],
    absolute: float, membrane: float, bending: float, shears: float, div: float,
    ) -> Tensor: ...
"""
Perform relaxation iterations, with local weighting

Parameters
----------
sol : (*batch, *spatial, ndim) tensor
hes : (*batch, *spatial, (ndim*(ndim+1))//2) tensor
grd : (*batch, *spatial, ndim) tensor
wgt : (*batch, *spatial) tensor
niter : int
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : float
membrane : float
bending : float
shears : float
div : float

Returns
-------
sol : (*batch, *spatial, ndim) tensor
"""

# ======================================================================

def field_matvec(
    out: Tensor, inp: Tensor, bound: list[int], voxel_size: list[float],
    absolute: list[float], membrane: list[float], bending: list[float],
    op: str = '=') -> Tensor: ...
"""
Forward matrix-vector product

Parameters
----------
out : (*batch, *spatial, nc) tensor
inp : (*batch, *spatial, nc) tensor
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : (nc,) list[float]
membrane : (nc,) list[float]
bending : (nc,) list[float]
op : {'=', '+', '-'}

Returns
-------
out : (*batch, *spatial, nc) tensor
"""

def field_matvec_rls(
    out: Tensor, inp: Tensor, wgt: Tensor, bound: list[int], voxel_size: list[float],
    absolute: list[float], membrane: list[float], bending: list[float],
    op: str = '=') -> Tensor: ...
"""
Forward matrix-vector product, with local weighting

Parameters
----------
out : (*batch, *spatial, nc) tensor
inp : (*batch, *spatial, nc) tensor
wgt : (*batch, *spatial, nc|1) tensor
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : (nc,) list[float]
membrane : (nc,) list[float]
bending : (nc,) list[float]
op : {'=', '+', '-'}

Returns
-------
out : (*batch, *spatial, nc) tensor
"""

def field_kernel(
    out: Tensor, bound: list[int], voxel_size: list[float],
    absolute: list[float], membrane: list[float], bending: list[float],
    op: str = '=') -> Tensor: ...
"""
Equivalent convolution kernel

Parameters
----------
out : (*batch, *spatial, nc) tensor
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : (nc,) list[float]
membrane : (nc,) list[float]
bending : (nc,) list[float]
op : {'=', '+', '-'}

Returns
-------
out : (*batch, *spatial, nc) tensor
"""

def field_diag(
    out: Tensor, bound: list[int], voxel_size: list[float],
    absolute: list[float], membrane: list[float], bending: list[float],
    op: str = '=') -> Tensor: ...
"""
Diagonal of the forward matrix

Parameters
----------
out : (*batch, *spatial, nc) tensor
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : (nc,) list[float]
membrane : (nc,) list[float]
bending : (nc,) list[float]
op : {'=', '+', '-'}

Returns
-------
out : (*batch, *spatial, nc) tensor
"""

def field_diag_rls(
    out: Tensor, wgt: Tensor, bound: list[int], voxel_size: list[float],
    absolute: list[float], membrane: list[float], bending: list[float],
    op: str = '=') -> Tensor: ...
"""
Diagonal of the forward matrix, with local weighting

Parameters
----------
out : (*batch, *spatial, nc) tensor
wgt : (*batch, *spatial, nc|1) tensor
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : (nc,) list[float]
membrane : (nc,) list[float]
bending : (nc,) list[float]
op : {'=', '+', '-'}

Returns
-------
out : (*batch, *spatial, nc) tensor
"""

def field_relax_(
    sol: Tensor, hes: Tensor, grd: Tensor, niter: int,
    bound: list[int], voxel_size: list[float],
    absolute: list[float], membrane: list[float], bending: list[float],
    ) -> Tensor: ...
"""
Perform relaxation iterations

Parameters
----------
sol : (*batch, *spatial, nc) tensor
hes : (*batch, *spatial, (nc*(nc+1))//2) tensor
grd : (*batch, *spatial, nc) tensor
niter : int
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : (nc,) list[float]
membrane : (nc,) list[float]
bending : (nc,) list[float]

Returns
-------
sol : (*batch, *spatial, nc) tensor
"""

def field_relax_rls_(
    sol: Tensor, hes: Tensor, grd: Tensor, wgt: Tensor, niter: int,
    bound: list[int], voxel_size: list[float],
    absolute: list[float], membrane: list[float], bending: list[float],
    ) -> Tensor: ...
"""
Perform relaxation iterations, with local weighting

Parameters
----------
sol : (*batch, *spatial, nc) tensor
hes : (*batch, *spatial, (nc*(nc+1))//2) tensor
grd : (*batch, *spatial, nc) tensor
wgt : (*batch, *spatial, nc|1) tensor
niter : int
bound : (ndim,) list[int]
voxel_size : (ndim,) list[float]
absolute : (nc,) list[float]
membrane : (nc,) list[float]
bending : (nc,) list[float]

Returns
-------
sol : (*batch, *spatial, nc) tensor
"""
