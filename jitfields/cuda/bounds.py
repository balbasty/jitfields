import os
from .atomic import code as atomic_code

code = ''
code += atomic_code + '\n'

this_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_folder, 'bounds.cuh'), 'rt') as f:
    code += f.read() + '\n'

convert_bound = {
    'zero': 0, 'zeros': 0,
    'repeat': 1, 'replicate': 1, 'nearest': 1,
    'dct1': 2, 'mirror': 2,
    'dct2': 3, 'reflect': 3,
    'dst1': 4, 'antimirror': 4,
    'dst2': 5, 'antireflect': 5,
    'dft': 6, 'warp': 6, 'circular': 6,
    'nocheck': 7,
}