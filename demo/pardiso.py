import numpy as np
from scipy import sparse as sp
import pypardiso
from jitfields.regularization import grid_kernel
from jitfields.utils import prod

import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt


def make_matrix(shape, **prm):
    ndim = len(shape)
    assert ndim == 3

    c = 2 if prm.get('bending', 0) else 1
    s = 5 if prm.get('bending', 0) else 3

    kernel = grid_kernel([s]*ndim, **prm).numpy()
    numel = prod(shape)

    mat = [[sp.lil_matrix((numel, numel), dtype=np.float32)
            for _ in range(ndim)] for _ in range(ndim)]

    i222 = np.arange(numel)
    i022 = np.roll(np.arange(numel).reshape(shape), -2, 0).flatten()
    i122 = np.roll(np.arange(numel).reshape(shape), -1, 0).flatten()
    i322 = np.roll(np.arange(numel).reshape(shape), +1, 0).flatten()
    i422 = np.roll(np.arange(numel).reshape(shape), +2, 0).flatten()
    i202 = np.roll(np.arange(numel).reshape(shape), -2, 1).flatten()
    i212 = np.roll(np.arange(numel).reshape(shape), -1, 1).flatten()
    i232 = np.roll(np.arange(numel).reshape(shape), +1, 1).flatten()
    i242 = np.roll(np.arange(numel).reshape(shape), +2, 1).flatten()
    i220 = np.roll(np.arange(numel).reshape(shape), -2, 2).flatten()
    i221 = np.roll(np.arange(numel).reshape(shape), -1, 2).flatten()
    i223 = np.roll(np.arange(numel).reshape(shape), +1, 2).flatten()
    i224 = np.roll(np.arange(numel).reshape(shape), +2, 2).flatten()
    i112 = np.roll(np.roll(np.arange(numel).reshape(shape), -1, 0), -1, 1).flatten()
    i132 = np.roll(np.roll(np.arange(numel).reshape(shape), -1, 0), +1, 1).flatten()
    i312 = np.roll(np.roll(np.arange(numel).reshape(shape), +1, 0), -1, 1).flatten()
    i332 = np.roll(np.roll(np.arange(numel).reshape(shape), +1, 0), +1, 1).flatten()
    i121 = np.roll(np.roll(np.arange(numel).reshape(shape), -1, 0), -1, 2).flatten()
    i123 = np.roll(np.roll(np.arange(numel).reshape(shape), -1, 0), +1, 2).flatten()
    i321 = np.roll(np.roll(np.arange(numel).reshape(shape), +1, 0), -1, 2).flatten()
    i323 = np.roll(np.roll(np.arange(numel).reshape(shape), +1, 0), +1, 2).flatten()
    i211 = np.roll(np.roll(np.arange(numel).reshape(shape), -1, 1), -1, 2).flatten()
    i213 = np.roll(np.roll(np.arange(numel).reshape(shape), -1, 1), +1, 2).flatten()
    i231 = np.roll(np.roll(np.arange(numel).reshape(shape), +1, 1), -1, 2).flatten()
    i233 = np.roll(np.roll(np.arange(numel).reshape(shape), +1, 1), +1, 2).flatten()

    if prm.get('shears', 0) or prm.get('div', 0):
        kernel_diag = kernel[..., np.arange(3), np.arange(3)]
    else:
        kernel_diag = kernel

    for d1 in range(ndim):
        mat[d1][d1][i222, i222] = kernel_diag[c, c, c, d1]
    if prm.get('membrane', 0) or prm.get('bending', 0) or prm.get('shears', 0) or prm.get('div', 0):
        for d1 in range(ndim):
            mat[d1][d1][i122, i222] = mat[d1][d1][i222, i122] = kernel_diag[c-1, c, c, d1]
            mat[d1][d1][i322, i222] = mat[d1][d1][i222, i322] = kernel_diag[c+1, c, c, d1]
            mat[d1][d1][i212, i222] = mat[d1][d1][i222, i212] = kernel_diag[c, c-1, c, d1]
            mat[d1][d1][i232, i222] = mat[d1][d1][i222, i232] = kernel_diag[c, c+1, c, d1]
            mat[d1][d1][i221, i222] = mat[d1][d1][i222, i221] = kernel_diag[c, c, c-1, d1]
            mat[d1][d1][i223, i222] = mat[d1][d1][i222, i223] = kernel_diag[c, c, c+1, d1]
    if prm.get('bending', 0):
        for d1 in range(ndim):
            mat[d1][d1][i022, i222] = mat[d1][d1][i222, i022] = kernel_diag[c-2, c, c, d1]
            mat[d1][d1][i422, i222] = mat[d1][d1][i222, i422] = kernel_diag[c+2, c, c, d1]
            mat[d1][d1][i202, i222] = mat[d1][d1][i222, i202] = kernel_diag[c, c-2, c, d1]
            mat[d1][d1][i242, i222] = mat[d1][d1][i222, i242] = kernel_diag[c, c+2, c, d1]
            mat[d1][d1][i220, i222] = mat[d1][d1][i222, i220] = kernel_diag[c, c, c-2, d1]
            mat[d1][d1][i224, i222] = mat[d1][d1][i222, i224] = kernel_diag[c, c, c+2, d1]

    if prm.get('shears', 0) or prm.get('div', 0):
        # off-diagonal
        for d1 in range(ndim):
            for d2 in range(ndim):
                if d2 == d1: continue
                mat[d1][d2][i222, i222] = kernel[c, c, c, d1, d2]
                mat[d1][d2][i112, i222] = mat[d1][d2][i222, i112] = kernel[c-1, c-1, c, d1, d2]
                mat[d1][d2][i132, i222] = mat[d1][d2][i222, i132] = kernel[c-1, c+1, c, d1, d2]
                mat[d1][d2][i312, i222] = mat[d1][d2][i222, i312] = kernel[c+1, c-1, c, d1, d2]
                mat[d1][d2][i332, i222] = mat[d1][d2][i222, i332] = kernel[c+1, c+1, c, d1, d2]
                mat[d1][d2][i121, i222] = mat[d1][d2][i222, i121] = kernel[c-1, c, c-1, d1, d2]
                mat[d1][d2][i123, i222] = mat[d1][d2][i222, i123] = kernel[c-1, c, c+1, d1, d2]
                mat[d1][d2][i321, i222] = mat[d1][d2][i222, i321] = kernel[c+1, c, c-1, d1, d2]
                mat[d1][d2][i323, i222] = mat[d1][d2][i222, i323] = kernel[c+1, c, c+1, d1, d2]
                mat[d1][d2][i211, i222] = mat[d1][d2][i222, i211] = kernel[c, c-1, c-1, d1, d2]
                mat[d1][d2][i213, i222] = mat[d1][d2][i222, i213] = kernel[c, c-1, c+1, d1, d2]
                mat[d1][d2][i231, i222] = mat[d1][d2][i222, i231] = kernel[c, c+1, c-1, d1, d2]
                mat[d1][d2][i233, i222] = mat[d1][d2][i222, i233] = kernel[c, c+1, c+1, d1, d2]

    mat = sp.vstack([sp.hstack(row) for row in mat])
    mat = sp.csr_matrix(mat)
    return mat


ndim = 3
# shape = [8] * ndim
shape = [192] * ndim

M = make_matrix(shape, absolute=0, shears=1, div=1)

# plt.imshow(M.todense())
# plt.colorbar()
# plt.show()

foo = 0
