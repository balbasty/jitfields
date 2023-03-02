import torch
from jitfields.solvers import ConjugateGradient, MultiGrid
from jitfields.resize import Prolong, Restrict
from jitfields.regularization import field_forward, field_precond
from jitfields.cpp.threads import set_num_threads

set_num_threads(1)

ndim = 3
shape = [192] * ndim
nc = 3

g = torch.zeros([*shape, nc])
g[64, 64, 64, :] = 5 * 1e3   # +64
g[64, 128, 64, :] = 5 * 1e3  # +64
g[128, 128, 64, :] = 5 * 1e3  # - 64
g[128, 64, 64, :] = 5 * 1e3  # -64

h = torch.zeros([*shape, (nc*(nc+1))//2])
h[64, 64, 64, :nc] = 1e3
h[64, 128, 64, :nc] = 1e3
h[128, 128, 64, :nc] = 1e3
h[128, 64, 64, :nc] = 1e3


def make_A(h, vx=1, membrane=1):
    def A(x, out=None):
        return field_forward(ndim, h, x, membrane=membrane, voxel_size=vx, out=out)
    return A


def make_P(h, vx=1, membrane=1):
    def P(x, out=None):
        return field_precond(ndim, h, x, membrane=membrane, voxel_size=vx, out=out)
    return P


# A = make_A(h)
# P = make_P(h)
# x = torch.zeros_like(g)
# x = ConjugateGradient(max_iter=32, tol=0).advanced_solve_(x, g, A, P)


import time
tic = time.time()
p = Prolong(ndim, order=2, channel_last=True)
r = Restrict(ndim, order=1, channel_last=True)
g = [g]
h = [h]
A = [make_A(h[0], 1)]
P = [make_P(h[0], 1)]
for k in range(1, 8):
    g += [r(g[-1])]
    h += [r(h[-1])]
    A += [make_A(h[-1], 2**k)]
    P += [make_P(h[-1], 2**k)]

x = torch.zeros_like(g[0])
x = MultiGrid(prolong=p, restrict=r).advanced_solve_(x, g, A, P)

toc = time.time()
print(toc - tic, 's')

foo = 0
