from jitfields.cpp.sym import (
    sym_matvec, sym_addmatvec_, sym_submatvec_, sym_solve, sym_solve_,
    sym_invert, sym_invert_
)
import torch

h = torch.as_tensor([1., 1., 0.1])
H = torch.as_tensor([[1., 0.1], [0.1, 1.]])
v = torch.randn([2])
o = torch.zeros([2])

print('\nmatvec')
print((H @ v[:, None]).flatten())
print(sym_matvec(o, v, h))
print(sym_addmatvec_(o.zero_(), v, h))
print(sym_submatvec_(o.zero_(), v, h).neg_())

print('\nsolve')
print((H.inverse() @ v[:, None]).flatten())
print(sym_solve_(o.copy_(v), h))
print(sym_solve(o, v, h))
print(sym_solve_(v, h))

print('\ninverse')
print(H.inverse())
print(sym_invert(h.clone(), h))
print(sym_invert_(h.clone()))
