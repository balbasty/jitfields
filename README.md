# jitfields
Fast functions for dense scalar and vector fields, implemented using just-in-time compilation.

**/!\ This is very experimental**

- I am implementing the GPU version of the algorithms in pure CUDA, which is compiled just-in-time by `cupy`.
- I am implementing the CPU version of the algorithms in pure C++, which gets just-in-time compiled by `cppyy`. 

Note that currently, the CPU implementation is single-threaded. 
I do plan to implement a multi-threaded parallel loop in the near future.

## Implemented so far

### Known bugs

- `spline_coeff` segfaults when the input shape is too small compared 
  to the spline order

### CPU and GPU

```python
euclidean_distance_transform(x, dim=None, vx=1, dtype=None)
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
```

```python
l1_distance_transform(x, dim=None, vx=1, dtype=None)
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
```

```python
resize(x, factor=None, shape=None, ndim=None,
       anchor='e', order=2, bound='dct2', prefilter=True)
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

References
----------
..[1]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part I-Theory,"
       IEEE Transactions on Signal Processing 41(2):821-832 (1993).
..[2]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
       IEEE Transactions on Signal Processing 41(2):834-848 (1993).
..[3]  M. Unser.
       "Splines: A Perfect Fit for Signal and Image Processing,"
       IEEE Signal Processing Magazine 16(6):22-38 (1999).
"""
```

```python
spline_coeff(inp, order, bound='dct2', dim=-1)
"""Compute the interpolating spline coefficients, along a single dimension.

Parameters
----------
inp : tensor
    Input tensor
order : {0..7}, default=2
    Interpolation order.
bound : {'zero', 'replicate', 'dct1', 'dct2', 'dft'}, default='dct2'
    Boundary conditions.
dim : int, default=-1
    Dimension along which to filter
    
Returns
-------
coeff : tensor
    Spline coefficients

References
----------
..[1]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part I-Theory,"
       IEEE Transactions on Signal Processing 41(2):821-832 (1993).
..[2]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
       IEEE Transactions on Signal Processing 41(2):834-848 (1993).
..[3]  M. Unser.
       "Splines: A Perfect Fit for Signal and Image Processing,"
       IEEE Signal Processing Magazine 16(6):22-38 (1999).
"""
```

```python
spline_coeff_nd(inp, order, bound='dct2', ndim=None)
"""Compute the interpolating spline coefficients, along the last N dimensions.

Parameters
----------
inp : (..., *spatial) tensor
    Input tensor
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dft'}, default='dct2'
    Boundary conditions.
ndim : int, default=`inp.dim()`
    Number of spatial dimensions
    
Returns
-------
coeff : (..., *spatial) tensor
    Spline coefficients

References
----------
..[1]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part I-Theory,"
       IEEE Transactions on Signal Processing 41(2):821-832 (1993).
..[2]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
       IEEE Transactions on Signal Processing 41(2):834-848 (1993).
..[3]  M. Unser.
       "Splines: A Perfect Fit for Signal and Image Processing,"
       IEEE Signal Processing Magazine 16(6):22-38 (1999).
"""
```

```python
restrict(x, factor=None, shape=None, ndim=None,
         anchor='e', order=2, bound='dct2', reduce_sum=False)
"""Restrict (adjoint of resize) a tensor using spline interpolation

Parameters
----------
x : (..., *inshape) tensor
    Input  tensor
factor : [sequence of] float, optional
    Factor by which to resize the tensor (> 1 == smaller)
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

Returns
-------
x : (..., *shape) tensor
    restricted tensor

"""
```
