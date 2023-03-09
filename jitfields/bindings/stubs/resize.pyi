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

Anchor methods
--------------
'e': Align exterior edges of the corner voxels
'c': Align centers of the corner voxels
'f': Align centers of the top-right corner, and use exact factor.
"""

def resize(out: Tensor, inp: Tensor, factor: list[float],
           anchor: str, order: list[int], bound: list[int]): ...
"""
Resize a volume encoded by B-splines

Parameters
----------
out : (..., *shape_out) Tensor
    Preallocated output placeholder
inp : (..., *shape_inp) Tensor
    Input volume to resize. 
    If order > 1, it should already contain spline coefficients.
factor : (ndim,) list[float] or None
    Resizing factor. Only required if `anchor == 'first'`,
anchor : {'edge', 'center', 'first'}
    Anchor points used for resizing.
order : (ndim,) list[0..7]
    Spline order, for each spatial dimension
bound : (ndim,) list[0..7]
    Boundary condition, for each spatial dimension
    
Returns
-------
out : (..., *shape_out) Tensor
    Resized volume.
"""
