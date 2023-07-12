# jitfields
Fast functions for dense scalar and vector fields, implemented using just-in-time compilation.

**/!\ This is (still) experimental**

- GPU version of the algorithms are written in pure CUDA, and compiled just-in-time by `cupy`.
- CPU version of the algorithms are written in pure C++, and compiled just-in-time by `cppyy`.

## Installation

Installation through pip should work, although I don't know how robust the cupy/pytorch 
interaction is in term of cuda version.
```sh
pip install git+https://github.com/balbasty/jitfields
```

If you intend to run code on the GPU, specify the [cuda] extra tag, which
ensures that cupy gets installed.
```sh
pip install git+https://github.com/balbasty/jitfields#egg=jitfields[cuda]
```

Pre-installing dependencies from conda may be more robust:
```sh
conda install -c python>=3.6 conda-forge numpy cupy ccpyy pytorch>=1.8 cudatoolkit=10.2
pip install git+https://github.com/balbasty/jitfields#egg=jitfields[cuda]
```

## Implemented so far

### Distance transforms

#### Distance to binary masks

```python
def euclidean_distance_transform(x, ndim=None, vx=1, dtype=None): ...
"""Compute the Euclidean distance transform of a binary image

Parameters
----------
x : (..., *spatial) tensor
    Input tensor
ndim : int, default=`x.ndim`
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
def l1_distance_transform(x, ndim=None, vx=1, dtype=None): ...
"""Compute the L1 distance transform of a binary image

Parameters
----------
x : (..., *spatial) tensor
    Input tensor
dim : int, default=`x.ndim`
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
def signed_distance_transform(
    x: tensor,
    ndim: Optional[int] = None,
    vx: OneOrSeveral[float] = 1,
    dtype: Optional[torch.dtype] = None,
) -> tensor: ...
"""Compute the signed Euclidean distance transform of a binary image

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
    Signed distance map, with shape `(..., *spatial)`.

References
----------
..[1] "Distance Transforms of Sampled Functions"
      Pedro F. Felzenszwalb & Daniel P. Huttenlocher
      Theory of Computing (2012)
      https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
"""
```

#### Distance to 1D splines

```python
def spline_distance_table(
    loc: tensor, 
    coeff: tensor, 
    steps: Optional[Union[int, tensor]] = None, 
    order: OrderType = 3, 
    bound: BoundType = 'dct2', 
    square: bool = False,
) -> Tuple[tensor, tensor]: ...
"""Compute the minimum distance from a set of points to a 1D spline

Parameters
----------
loc : `(..., D) tensor`
    Point set.
coeff : `(..., N, D) tensor`
    Spline coefficients encoding the location of the 1D spline.
steps : `int or (..., K) tensor`
    Number of time steps to try, or list of time steps to try.
order : {1..7}
    Spline order.
bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`
    Boundary conditions of the spline.
square : bool
    Return the squared Euclidean distance.

Returns
-------
dist : `(...) tensor`
    Distance from each point in the set to its closest point on the spline
time : `(...) tensor`
    Time of the closest point on the spline
"""
```

```python
def spline_distance_brent(
    loc: tensor, 
    coeff: tensor, 
    max_iter: int = 128, 
    tol: float = 1e-6, 
    step_size: float = 0.01, 
    order: OrderType = 3, 
    bound: BoundType = 'dct2', 
    square: bool = False,
    steps: Optional[Union[int, tensor]] = None, 
) -> Tuple[tensor, tensor]: ...
"""Compute the minimum distance from a set of points to a 1D spline

Parameters
----------
loc : `(..., D) tensor`
    Point set.
coeff : `(..., N, D) tensor`
    Spline coefficients encoding the location of the 1D spline.
max_iter : int
    Number of optimization steps.
tol : float
    Tolerance for early stopping
step_size : float
    Initial search size.
order : {1..7}
    Spline order.
bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`
    Boundary conditions of the spline.
square : bool
    Return the squared Euclidean distance.
steps : int
    Number of steps used in the table-based initialisation.

Returns
-------
dist : `(...) tensor`
    Distance from each point in the set to its closest point on the spline
time : `(...) tensor`
    Time of the closest point on the spline
"""
```

```python
def spline_distance_gaussnewton(
    loc: tensor, 
    coeff: tensor, 
    max_iter: int = 16, 
    tol: float = 1e-6, 
    order: OrderType = 3, 
    bound: BoundType = 'dct2', 
    square: bool = False,
    steps: Optional[Union[int, tensor]] = None, 
) -> Tuple[tensor, tensor]: ...
"""Compute the minimum distance from a set of points to a 1D spline

Parameters
----------
loc : `(..., D) tensor`
    Point set.
coeff : `(..., N, D) tensor`
    Spline coefficients encoding the location of the 1D spline.
max_iter : int
    Number of optimization steps.
tol : float
    Tolerance for early stopping
order : {1..7}
    Spline order.
bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`
    Boundary conditions of the spline.
square : bool
    Return the squared Euclidean distance.
steps : int
    Number of steps used in the table-based initialisation.

Returns
-------
dist : `(...) tensor`
    Distance from each point in the set to its closest point on the spline
time : `(...) tensor`
    Time of the closest point on the spline
"""
```

```python
def spline_distance_brent_(
    dist: tensor, 
    time: tensor, 
    loc: tensor, 
    coeff: tensor, 
    max_iter: int = 128, 
    tol: float = 1e-6, 
    step_size: float = 0.01, 
    order: OrderType = 3, 
    bound: BoundType = 'dct2', 
    square: bool = False,
) -> Tuple[tensor, tensor]: ...
"""Compute the minimum distance from a set of points to a 1D spline (inplace)

Parameters
----------
dist : `(...) tensor`
    Initial distance from each point in the set to its closest point on the spline
time : `(...) tensor`
    Initial time of the closest point on the spline
loc : `(..., D) tensor`
    Point set.
coeff : `(..., N, D) tensor`
    Spline coefficients encoding the location of the 1D spline.
max_iter : int
    Number of optimization steps.
tol : float
    Tolerance for early stopping
step_size : float
    Initial search size.
order : {1..7}
    Spline order.
bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`
    Boundary conditions of the spline.
square : bool
    Return the squared Euclidean distance.

Returns
-------
dist : `(...) tensor`
    Distance from each point in the set to its closest point on the spline
time : `(...) tensor`
    Time of the closest point on the spline
"""
```

```python
def spline_distance_gaussnewton_(
    dist: tensor, 
    time: tensor, 
    loc: tensor, 
    coeff: tensor, 
    max_iter: int = 16, 
    tol: float = 1e-6, 
    order: OrderType = 3, 
    bound: BoundType = 'dct2', 
    square: bool = False,
) -> Tuple[tensor, tensor]: ...
"""Compute the minimum distance from a set of points to a 1D spline (inplace)

Parameters
----------
dist : `(...) tensor`
    Initial distance from each point in the set to its closest point on the spline
time : `(...) tensor`
    Initial time of the closest point on the spline
loc : `(..., D) tensor`
    Point set.
coeff : `(..., N, D) tensor`
    Spline coefficients encoding the location of the 1D spline.
max_iter : int
    Number of optimization steps.
tol : float
    Tolerance for early stopping
order : {1..7}
    Spline order.
bound : `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`
    Boundary conditions of the spline.
square : bool
    Return the squared Euclidean distance.

Returns
-------
dist : `(...) tensor`
    Distance from each point in the set to its closest point on the spline
time : `(...) tensor`
    Time of the closest point on the spline
"""
```

#### Distance to triangular meshes

```python
def mesh_distance_signed(
    loc: tensor, 
    vertices: tensor, 
    faces: tensor,
    out: Optional[tensor] = None,
) -> tensor: ...
"""Compute the *signed* minimum distance from a set of points to a triangular mesh

Parameters
----------
loc : `(..., D) tensor`
    Point set.
vertices : `(N, D) tensor`
    Mesh vertices
faces : `(M, D) tensor[integer]`
    Mesh faces

Returns
-------
dist : `(...) tensor`
    Signed distance from each point in the set to its closest point on the mesh
    (negative inside, positive outside)
"""
```

```python
def mesh_distance(
    loc: tensor, 
    vertices: tensor, 
    faces: tensor,
    out: Optional[tensor] = None,
) -> tensor: ...
"""Compute the minimum distance from a set of points to a triangular mesh

Parameters
----------
loc : `(..., D) tensor`
    Point set.
vertices : `(N, D) tensor`
    Mesh vertices
faces : `(M, D) tensor[integer]`
    Mesh faces

Returns
-------
dist : `(...) tensor`
    Signed distance from each point in the set to its closest point on the mesh
    (negative inside, positive outside)
"""
```

### Interpolation/Resampling

```python
def spline_coeff(inp, order, bound='dct2', dim=-1): ...
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
def spline_coeff_nd(inp, order, bound='dct2', ndim=None): ...
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
def resize(x, factor=None, shape=None, ndim=None,
           anchor='e', order=2, bound='dct2', prefilter=True): ...
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
def restrict(x, factor=None, shape=None, ndim=None,
             anchor='e', order=2, bound='dct2', reduce_sum=False): ...
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
def pull(inp, grid, order=2, bound='dct2', extrapolate=True, prefilter=False, out=None): ...
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
def push(inp, grid, shape=None, order=2, bound='dct2', extrapolate=True, prefilter=False, out=None): ...
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
def count(grid, shape=None, order=2, bound='dct2', extrapolate=True, out=None): ...
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
def grad(inp, grid, order=2, bound='dct2', extrapolate=True, prefilter=False, out=None): ...
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
def sym_matvec(mat, vec, dtype=None, out=None): ...
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
def sym_addmatvec(inp, mat, vec, dtype=None, out=None): ...
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
def sym_addmatvec_(inp, mat, vec, dtype=None): ...
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
def sym_submatvec(inp, mat, vec, dtype=None, out=None): ...
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
def sym_submatvec_(inp, mat, vec, dtype=None): ...
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
def sym_solve(mat, vec, dtype=None, out=None): ...
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
def sym_solve_(mat, vec, dtype=None): ...
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
def sym_invert(mat, dtype=None, out=None): ...
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
def sym_invert_(mat, dtype=None): ...
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

### Regularisers for dense flow fields

```python
def flow_matvec(
    vel: Tensor, weight: Optional[Tensor] = None,
    absolute: float = 0, membrane: float = 0, bending: float = 0,
    shears: float = 0, div: float = 0,
    bound: list[str] = 'dft', voxel_size: list[float] = 1,
    out: Optional[Tensor] = None) -> Tensor: ...
"""Apply a spatial regularization matrix.

Parameters
----------
vel : (*batch, *spatial, ndim) tensor
    Input displacement field, in voxels.
weight : (*batch, *spatial) tensor, optional
    Weight map, to spatially modulate the regularization.
absolute : float
    Penalty on absolute values.
membrane : float
    Penalty on first derivatives.
bending : float
    Penalty on second derivatives.
shears : float
    Penalty on local shears.
div : float
    Penalty on local volume changes.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
    Boundary conditions.
voxel_size : [sequence of] float
    Voxel size.
out : (*batch, *spatial, ndim) tensor, optional
    Output placeholder

Returns
-------
out : (*batch, *spatial, ndim) tensor
"""

# We also implement variants that adds to or subtracts from an input tensor
def flow_matvec_add(inp: Tensor, ...): ...
def flow_matvec_add_(inp: Tensor, ...): ...
def flow_matvec_sub(inp: Tensor, ...): ...
def flow_matvec_sub_(inp: Tensor, ...): ...
```

```python
def flow_kernel(
    shape: list[int],
    absolute: float = 0, membrane: float = 0, bending: float = 0,
    shears: float = 0, div: float = 0,
    bound: list[str] = 'dft', voxel_size: list[float] = 1,
    out: Optional[Tensor] = None) -> Tensor: ...
"""
Return the kernel of a Toeplitz regularization matrix.

Parameters
----------
shape : int or list[int]
    Number of spatial dimensions or shape of the tensor
absolute : float
    Penalty on absolute values.
membrane : float
    Penalty on first derivatives.
bending : float
    Penalty on second derivatives.
shears : float
    Penalty on local shears. Linear elastic energy's `mu`.
div : float
    Penalty on local volume changes. Linear elastic energy's `lambda`.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
    Boundary conditions.
voxel_size : [sequence of] float
    Voxel size.
out : (*shape, ndim, [ndim]) tensor, optional
    Output placeholder

Returns
-------
out : (*shape, ndim, [ndim]) tensor
    Convolution kernel.
    A matrix or kernels ([ndim, ndim]) if `shears` or `div`,
    else a vector of kernels ([ndim]) .
"""

# We also implement variants that adds to or subtracts from an input tensor
def flow_kernel_add(inp: Tensor, ...): ...
def flow_kernel_add_(inp: Tensor, ...): ...
def flow_kernel_sub(inp: Tensor, ...): ...
def flow_kernel_sub_(inp: Tensor, ...): ...
```

```python
def flow_diag(
    shape: list[int], weight: Optional[Tensor] = None,
    absolute: float = 0, membrane: float = 0, bending: float = 0,
    shears: float = 0, div: float = 0,
    bound: list[str] = 'dft', voxel_size: list[float] = 1,
    out: Optional[Tensor] = None) -> Tensor: ...
"""Return the diagonal of a regularization matrix.

Parameters
----------
shape : list[int]
    Shape of the tensor
weight : (*batch, *spatial) tensor, optional
    Weight map, to spatially modulate the regularization.
absolute : float
    Penalty on absolute values.
membrane : float
    Penalty on first derivatives.
bending : float
    Penalty on second derivatives.
shears : float
    Penalty on local shears.
div : float
    Penalty on local volume changes.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
    Boundary conditions.
voxel_size : [sequence of] float
    Voxel size.
out : (*batch, *spatial, ndim) tensor, optional
    Output placeholder

Returns
-------
out : (*batch, *spatial, ndim) tensor
"""

# We also implement variants that adds to or subtracts from an input tensor
def flow_diag_add(inp: Tensor, ...): ...
def flow_diag_add_(inp: Tensor, ...): ...
def flow_diag_sub(inp: Tensor, ...): ...
def flow_diag_sub_(inp: Tensor, ...): ...
```

```python
def flow_relax_(
    vel: Tensor, hes: Tensor, grd: Tensor, weight: Optional[Tensor] = None,
    absolute: float = 0, membrane: float = 0, bending: float = 0,
    shears: float = 0, div: float = 0,
    bound: list[str] = 'dft', voxel_size: list[float] = 1, nb_iter: int = 1,
    ) -> Tensor: ...
"""Perform relaxation iterations.

Parameters
----------
vel : (*batch, *spatial, ndim) tensor
    Warm start.
hes : (*batch, *spatial, ndim*(ndim+1)//2) tensor
    Input symmetric Hessian, in voxels.
grd : (*batch, *spatial, ndim) tensor
    Input gradient, in voxels.
weight : (*batch, *spatial) tensor, optional
    Weight map, to spatially modulate the regularization.
absolute : float
    Penalty on absolute values.
membrane : float
    Penalty on first derivatives.
bending : float
    Penalty on second derivatives.
shears : float
    Penalty on local shears.
div : float
    Penalty on local volume changes.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
    Boundary conditions.
voxel_size : [sequence of] float
    Voxel size.
nb_iter : int
    Number of iterations

Returns
-------
vel : (*batch, *spatial, ndim) tensor
"""
```

```python

def flow_precond(
    mat: Tensor, vec: Tensor, weight : Optional[Tensor] = None,
    absolute: float = 0, membrane: float = 0, bending: float = 0,
    shears: float = 0, div: float = 0,
    bound: list[str] = 'dft', voxel_size: list[float] = 1,
    out: Optional[Tensor] = None) -> Tensor: ...
"""
Apply the preconditioning `(M + diag(R)) \ v`

Parameters
----------
mat : (*batch, *spatial, DD) tensor
    DD == 1 | D | D*(D+1)//2 | D*D
    Preconditioning matrix `M`
vec : (*batch, *spatial, D) tensor
    Point `v` at which to solve the system.
weight : (*batch, *spatial) tensor, optional
    Regularization weight map.
absolute : float
    Penalty on absolute values.
membrane : float
    Penalty on first derivatives.
bending : float
    Penalty on second derivatives.
shears : float
    Penalty on local shears.
div : float
    Penalty on local volume changes.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
    Boundary conditions.
voxel_size : [sequence of] float
    Voxel size.
out : (*batch, *spatial, D) tensor
    Output placeholder.

Returns
-------
out : (*batch, *spatial, D) tensor
    Preconditioned vector.

"""
```

```python
def flow_forward(
    mat: Tensor, vec: Tensor, weight : Optional[Tensor] = None,
    absolute: float = 0, membrane: float = 0, bending: float = 0,
    shears: float = 0, div: float = 0,
    bound: list[str] = 'dft', voxel_size: list[float] = 1,
    out: Optional[Tensor] = None) -> Tensor: ...
"""
Apply the forward matrix-vector product `(M + R) @ v`

Parameters
----------
mat : (*batch, *spatial, DD) tensor
    DD == 1 | D | D*(D+1)//2 | D*D
vec : (*batch, *spatial, D) tensor
    Point `v` at which to solve the system.
weight : (*batch, *spatial) tensor, optional
    Regularization weight map.
absolute : float
    Penalty on absolute values.
membrane : float
    Penalty on first derivatives.
bending : float
    Penalty on second derivatives.
shears : float
    Penalty on local shears.
div : float
    Penalty on local volume changes.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dft'
    Boundary conditions.
voxel_size : [sequence of] float
    Voxel size.
out : (*batch, *spatial, D) tensor
    Output placeholder.

Returns
-------
out : (*batch, *spatial, D) tensor
    Preconditioned vector.

"""
```