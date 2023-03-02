# jitfields
Fast functions for dense scalar and vector fields, implemented using just-in-time compilation.

**/!\ This is very experimental**

- GPU version of the algorithms are written in pure CUDA, and compiled just-in-time by `cupy`.
- CPU version of the algorithms are written in pure C++, and compiled just-in-time by `cppyy`.

## Recent changes
- While most CPU algorithms are single-threaded, I have added a thread pool
and basic parallel loop, which is in use in a few selected functions.
However, the thread pool only works on LLVM >= 13, which is not the default
version on some linux distributions. I plan to add another parallel 
implementation using OpenMP. I hope that it'll be possible to select the 
most appropriate backend at startup.

## Installation

Installation through pip should work, although I don't know how robust the cupy/pytorch 
interaction is in term of cuda version.
```sh
pip install git+https://github.com/balbasty/jitfields
```

Pre-installing dependencies from conda may be more robust:
```sh
conda install -c conda-forge numpy cupy ccpyy pytorch cudatoolkit=10.2
pip install git+https://github.com/balbasty/jitfields
```

## Implemented so far

### Distance transforms

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


### Interpolation/Resampling

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

```python
pull(inp, grid, order=2, bound='dct2', extrapolate=True, prefilter=False, out=None)
"""Sample a tensor using spline interpolation

Parameters
----------
inp : (..., *inshape, channel) tensor
    Input tensor
grid : (..., *outshape, ndim) tensor
    Tensor of coordinates into `inp`
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
    How to deal with out-of-bound values.
extrapolate : bool or {'center', 'edge'}
    - True: use bound to extrapolate out-of-bound value
    - False or 'center': do not extrapolate values that fall outside
      of the centers of the first and last voxels.
    - 'edge': do not extrapolate values that fall outside
       of the edges of the first and last voxels.
prefilter : bool, default=True
    Whether to first compute interpolating coefficients.
    Must be true for proper interpolation, otherwise this
    function merely performs a non-interpolating "spline sampling".

Returns
-------
out : (..., *outshape, channel) tensor
    Pulled tensor

"""
```

```python
push(inp, grid, shape=None, order=2, bound='dct2', extrapolate=True, prefilter=False, out=None)
"""Splat a tensor using spline interpolation

Parameters
----------
inp : (..., *inshape, channel) tensor
    Input tensor
grid : (..., *inshape, ndim) tensor
    Tensor of coordinates into `inp`
shape : sequence[int], default=inshape
    Output spatial shape
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
    How to deal with out-of-bound values.
extrapolate : bool or {'center', 'edge'}
    - True: use bound to extrapolate out-of-bound value
    - False or 'center': do not extrapolate values that fall outside
      of the centers of the first and last voxels.
    - 'edge': do not extrapolate values that fall outside
       of the edges of the first and last voxels.
    prefilter : bool, default=True
        Whether to compute interpolating coefficients at the end.

Returns
-------
out : (..., *shape, channel) tensor
    Pulled tensor
"""
```

```python
count(grid, shape=None, order=2, bound='dct2', extrapolate=True, out=None)
"""Splat ones using spline interpolation

Parameters
----------
grid : (..., *inshape, ndim) tensor
    Tensor of coordinates
shape : sequence[int], default=inshape
    Output spatial shape
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
    How to deal with out-of-bound values.
extrapolate : bool or {'center', 'edge'}
    - True: use bound to extrapolate out-of-bound value
    - False or 'center': do not extrapolate values that fall outside
      of the centers of the first and last voxels.
    - 'edge': do not extrapolate values that fall outside
       of the edges of the first and last voxels.

Returns
-------
out : (..., *shape) tensor
    Pulled tensor
"""
```

```python
grad(inp, grid, order=2, bound='dct2', extrapolate=True, prefilter=False, out=None)
"""Sample the spatial gradients of a tensor using spline interpolation

Parameters
----------
inp : (..., *inshape, channel) tensor
    Input tensor
grid : (..., *outshape, ndim) tensor
    Tensor of coordinates into `inp`
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
    How to deal with out-of-bound values.
extrapolate : bool or {'center', 'edge'}
    - True: use bound to extrapolate out-of-bound value
    - False or 'center': do not extrapolate values that fall outside
      of the centers of the first and last voxels.
    - 'edge': do not extrapolate values that fall outside
       of the edges of the first and last voxels.
prefilter : bool, default=True
    Whether to first compute interpolating coefficients.
    Must be true for proper interpolation, otherwise this
    function merely performs a non-interpolating "spline sampling".

Returns
-------
out : (..., *outshape, channel, ndim) tensor
    Pulled gradients
"""
```

### Compact symmetric (or postive-definite) matrices

Currently only implemented on the CPU.

```python
sym_matvec(mat, vec, dtype=None, out=None)
"""Matrix-vector product for compact symmetric matrices

    `out = mat @ vec`

Parameters
----------
mat : (..., C*(C+1)//2) tensor
    Symmetric matrix with compact storage.
    The matrix should be saved as a vector containing the diagonal
    followed by the rows of the upper triangle.
vec : (..., C) tensor
    Vector
dtype : torch.dtype, optional
    Data type used to carry the computation. By default, same as input.
out : (..., C) tensor, optional
    Output placeholder

Returns
-------
out : (..., C) tensor
    Matrix-vector product
"""
```

```python
sym_addmatvec(inp, mat, vec, dtype=None, out=None)
"""Add a matrix-vector product for compact symmetric matrices

    `out = inp + mat @ vec`

Parameters
----------
inp : (..., C) tensor
    Vector to which the matrix-vector product is added
mat : (..., C*(C+1)//2) tensor
    Symmetric matrix with compact storage.
    The matrix should be saved as a vector containing the diagonal
    followed by the rows of the upper triangle.
vec : (..., C) tensor
    Vector used in the matrix-vector product
dtype : torch.dtype, optional
    Data type used to carry the computation. By default, same as input.
out : (..., C) tensor, optional
    Output placeholder

Returns
-------
out : (..., C) tensor
    Added matrix-vector product
"""
```

```python
sym_addmatvec_(inp, mat, vec, dtype=None)
"""Inplace add a matrix-vector product for compact symmetric matrices

    `inp += mat @ vec`

Parameters
----------
inp : (..., C) tensor
    Vector to which the matrix-vector product is added
mat : (..., C*(C+1)//2) tensor
    Symmetric matrix with compact storage.
    The matrix should be saved as a vector containing the diagonal
    followed by the rows of the upper triangle.
vec : (..., C) tensor
    Vector used in the matrix-vector product
dtype : torch.dtype, optional
    Data type used to carry the computation. By default, same as input.

Returns
-------
inp : (..., C) tensor
    Added matrix-vector product
"""
```

```python
sym_submatvec(inp, mat, vec, dtype=None, out=None)
"""Subtract a matrix-vector product for compact symmetric matrices

    `out = inp - mat @ vec`

Parameters
----------
inp : (..., C) tensor
    Vector to which the matrix-vector product is added
mat : (..., C*(C+1)//2) tensor
    Symmetric matrix with compact storage.
    The matrix should be saved as a vector containing the diagonal
    followed by the rows of the upper triangle.
vec : (..., C) tensor
    Vector used in the matrix-vector product
dtype : torch.dtype, optional
    Data type used to carry the computation. By default, same as input.
out : (..., C) tensor, optional
    Output placeholder

Returns
-------
out : (..., C) tensor
    Subtracted matrix-vector product
"""
```

```python
sym_submatvec_(inp, mat, vec, dtype=None)
"""Inplace subtract a matrix-vector product for compact symmetric matrices

    `inp -= mat @ vec`

Parameters
----------
inp : (..., C) tensor
    Vector to which the matrix-vector product is added
mat : (..., C*(C+1)//2) tensor
    Symmetric matrix with compact storage.
    The matrix should be saved as a vector containing the diagonal
    followed by the rows of the upper triangle.
vec : (..., C) tensor
    Vector used in the matrix-vector product
dtype : torch.dtype, optional
    Data type used to carry the computation. By default, same as input.

Returns
-------
inp : (..., C) tensor
    Subtracted matrix-vector product
"""
```

```python
sym_solve(mat, vec, dtype=None, out=None)
"""Solve the symmetric linear system

    `out = mat.inverse() @ vec`

!! Does not backpropagate through `mat` !!

Parameters
----------
mat : (..., C*(C+1)//2) tensor
    Symmetric matrix with compact storage.
    The matrix should be saved as a vector containing the diagonal
    followed by the rows of the upper triangle.
vec : (..., C) tensor
    Vector
dtype : torch.dtype, optional
    Data type used to carry the computation. By default, same as input.
out : (..., C) tensor, optional
    Output placeholder

Returns
-------
out : (..., C) tensor
    Solution of the linear system
"""
```

```python
sym_solve_(mat, vec, dtype=None)
"""Solve the symmetric linear system in-place

    `vec = mat.inverse() @ vec`

!! Does not backpropagate through `mat` !!

Parameters
----------
mat : (..., C*(C+1)//2) tensor
    Symmetric matrix with compact storage.
    The matrix should be saved as a vector containing the diagonal
    followed by the rows of the upper triangle.
vec : (..., C) tensor
    Vector
dtype : torch.dtype, optional
    Data type used to carry the computation. By default, same as input.

Returns
-------
vec : (..., C) tensor
    Solution of the linear system
"""
```

```python
sym_invert(mat, dtype=None, out=None)
"""Invert a compact symmetric matrix

    `out = mat.inverse()`

!! Does not backpropagate through `mat` !!

Parameters
----------
mat : (..., C*(C+1)//2) tensor
    Symmetric matrix with compact storage.
    The matrix should be saved as a vector containing the diagonal
    followed by the rows of the upper triangle.
dtype : torch.dtype, optional
    Data type used to carry the computation. By default, same as input.
out : (..., C*(C+1)//2) tensor, optional
    Output placeholder

Returns
-------
mat : (..., C*(C+1)//2) tensor
    Inverse matrix

"""
```

```python
sym_invert_(mat, dtype=None)
"""Invert a compact symmetric matrix in-place

    `mat = mat.inverse()`

!! Does not backpropagate through `mat` !!

Parameters
----------
mat : (..., C*(C+1)//2) tensor
    Symmetric matrix with compact storage.
    The matrix should be saved as a vector containing the diagonal
    followed by the rows of the upper triangle.
dtype : torch.dtype, optional
    Data type used to carry the computation. By default, same as input.

Returns
-------
mat : (..., C*(C+1)//2) tensor
    Inverse matrix

"""
```
