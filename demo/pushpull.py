from jitfields.pushpull import pull, push, count, grad
from jitfields.resize import resize
from jitfields.utils import identity_grid, add_identity_grid_
import torch
import matplotlib.pyplot as plt

shape = [16, 16, 16]

grid = resize(2 * torch.randn([3, 4, 4, 4]), shape=shape, order=3).movedim(0, -1)

plt.imshow(grid.square().sum(-1).sqrt()[8])
plt.colorbar()
plt.show()

grid = add_identity_grid_(grid)

circle = identity_grid([16, 16, 16]).sub_(8)
circle = circle.square_().sum(-1).sqrt_()
circle = (circle < 6).float()
circle += torch.rand_like(circle).mul_(0.1)

plt.imshow(circle[8])
plt.colorbar()
plt.show()

circle.requires_grad_()
grid.requires_grad_()

wircle = pull(circle[..., None], grid, order=1)[..., 0]

plt.imshow(wircle.detach()[8])
plt.colorbar()
plt.show()
#
# gircle = grad(circle[..., None], grid)[..., 0, :]
#
# plt.imshow(gircle[..., 0].detach()[8])
# plt.colorbar()
# plt.show()
#
# plt.imshow(gircle[..., 1].detach()[8])
# plt.colorbar()
# plt.show()
#
# # plt.imshow(gircle[32, ..., 2].detach())
# # plt.colorbar()
# # plt.show()
#
# #%
#
# # wircle.requires_grad_()
# wircle.square().sum().backward()
# # print(wircle.grad.min(), wircle.grad.max())
#
# plt.imshow(circle.grad[8])
# plt.colorbar()
# plt.show()

# plt.imshow(grid.grad[..., 0][8])
# plt.colorbar()
# plt.show()

# print(circle.grad.min(), circle.grad.max())
# print(grid.grad.min(), grid.grad.max())
