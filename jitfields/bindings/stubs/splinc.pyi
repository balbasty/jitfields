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
"""

def spline_coeff_(inp: Tensor, order: int, bound: int, dim: int =-1
                  ) -> Tensor: ...
"""
Perform spline prefiltering along a dimension, inplace.

Parameters
----------
inp : (..., N) tensor
    Input volume
order : 0..7
    Spline order
bound : 0..7
    Boundary conditions
dim : int
    Dimension along which to prefilter
    
Returns
-------
out : (..., N) tensor
    Output volume of spine coefficients
"""

def spline_coeff_nd_(inp: Tensor, order: int, bound: int, ndim: int =-1
                     ) -> Tensor: ...
"""
Perform spline prefiltering along the last `N` dimensions, inplace.

Parameters
----------
inp : (..., *shape) tensor
    Input volume
order : (ndim,) list[0..7]
    Spline order along each spatial dimension
bound : (ndim,) list[0..7]
    Boundary conditions along each spatial dimension
dim : int
    Number of spatial dimensions

Returns
-------
out : (..., *shape) tensor
    Output volume of spine coefficients
"""
