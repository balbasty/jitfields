from typing import TypeVar, Union, Sequence, Optional

try:
    from typing import Literal
    LiteralInt = LiteralStr = LiteralStrOpt = Literal
except ImportError:

    class _LiteralInt:
        def __getitem__(self, _):
            return int

    class _LiteralStr:
        def __getitem__(self, _):
            return str

    class _LiteralStrOpt:
        def __getitem__(self, _):
            return Optional[str]

    LiteralInt = _LiteralInt()
    LiteralStr = _LiteralStr()
    LiteralStrOpt = _LiteralStrOpt()


T = TypeVar('T')

OneOrSeveral = Union[T, Sequence[T]]
"""A single scalar value or a sequence (list, tuple, ...) of scalar values"""

BoundType = LiteralStr[
    'zero', 'zeros',
    'replicate', 'border', 'nearest', 'repeat',
    'dct1', 'mirror',
    'dct2', 'reflect',
    'dst1', 'antimirror',
    'dst2', 'antireflect',
    'dft', 'wrap', 'circular', 'circulant',
]
"""
There is a lack of standardization for bound names across packages.
We mostly work with scipy's convention and/or with aliases that relate
to the implicit boundary conditions in discrete transforms (Fourier, sine
and cosine). We also accept less common aliases, although their use is
discouraged

| Fourier       | SciPy            | Other                                  | Description               |
| ------------- | ---------------- | -------------------------------------- | ------------------------- |
|               | `"border"`       | `"nearest"`, `"replicate"`, `"repeat"` | ` a  a | a b c d |  d  d` |
|               | <del>`"constant"`</del> | `"zero"`, `"zeros"`             | ` 0  0 | a b c d |  0  0` |
| `"dft"`       | `"wrap"`         | `"circular"`, `"circulant"`            | ` c  d | a b c d |  a  b` |
| `"dct1"`      | `"mirror"`       |                                        | ` c  b | a b c d |  c  b` |
| `"dct2"`      | `"reflect"`      |                                        | ` b  a | a b c d |  d  c` |
| `"dst1"`      |                  | `"antimirror"`                         | `-a  0 | a b c d |  0 -d` |
| `"dst2"`      |                  | `"antireflect"`                        | `-b -a | a b c d | -d -c` |

"""  # noqa: E501

OrderType = LiteralInt[0, 1, 2, 3, 4, 5, 6, 7]
"""Interpolation orders are integers in the range `{0..7}`"""

_ExtrapolateType = LiteralStr['center', 'edge']
ExtrapolateType = Union[bool, _ExtrapolateType]
"""Extrapolation can be a boolean or one of `{'center', 'edge'}`

- `True`: use bound to extrapolate out-of-bound value
- `False` or `'center'`: do not extrapolate values that fall outside
of the centers of the first and last voxels.
- `'edge'`: do not extrapolate values that fall outside
of the edges of the first and last voxels.
"""

AnchorType = LiteralStrOpt['center', 'edge', None]
"""
What feature should be aligned across the input and output tensors.

- If `'edge'`, align the exterior edges of the first and last voxels.
- If `'center'`, align the centers of the first and last voxels.
- If `None`, the center of the first voxel is aligned, and the
requested factor is exactly applied.

If `'edge'` or `'center'`, the effective scaling factor may slightly
differ from the requested scaling factor.
"""
