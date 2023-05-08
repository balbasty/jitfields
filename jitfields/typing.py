from typing import TypeVar, Union, Sequence, Literal

T = TypeVar('T')

OneOrSeveral = Union[T, Sequence[T]]
"""A single scalar value or a sequence (list, tuple, ...) of scalar values"""

BoundType = Literal['zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft']
"""A Bound can be one of `{'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`"""

OrderType = Literal[0, 1, 2, 3, 4, 5, 6, 7]
"""Interpolation orders are integers in the range `{0..7}`"""

ExtrapolateType = Union[bool, Literal['center', 'edge']]
"""Extrapolation can be a boolean or one of `{'center', 'edge'}`

- `True`: use bound to extrapolate out-of-bound value
- `False` or `'center'`: do not extrapolate values that fall outside
of the centers of the first and last voxels.
- `'edge'`: do not extrapolate values that fall outside
of the edges of the first and last voxels.
"""

AnchorType = Literal['center', 'edge', None]
"""
What feature should be aligned across the input and output tensors.

- If `'edge'`, align the exterior edges of the first and last voxels.
- If `'center'`, align the centers of the first and last voxels.
- If `None`, the center of the first voxel is aligned, and the
requested factor is exactly applied.

If `'edge'` or `'center'`, the effective scaling factor may slightly
differ from the requested scaling factor.
"""
