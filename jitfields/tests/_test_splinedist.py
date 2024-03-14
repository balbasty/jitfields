import torch
import matplotlib.pyplot as plt
from jitfields.distance import (
    spline_distance_table, 
    spline_distance_brent, 
    spline_distance_brent_, 
    spline_distance_gaussnewton,
    spline_distance_gaussnewton_,
)
from jitfields.pushpull import pull
from time import time as timer

coeff = torch.as_tensor([[5, 5], [32, 60], [60, 5]]).float()
grid = torch.stack(torch.meshgrid(torch.arange(64, dtype=torch.float32), 
                                  torch.arange(64, dtype=torch.float32)), -1)

values = pull(coeff, torch.linspace(0, 2, 512)[:, None], order=3, bound='dct2')

# grid = grid.cuda()
# coeff = coeff.cuda()

for steps in (8, 32, 64, 128, 256):

    tic = timer()
    dist, time = spline_distance_table(grid, coeff, steps=steps)
    toc = timer() - tic
    print(f'table ({steps})', toc * 1e3, 'ms')

    plt.subplot(2, 2, 1)
    plt.imshow(dist.cpu(), cmap=plt.get_cmap("viridis", 2**16), interpolation='nearest')
    plt.plot(values[:, 1], values.numpy()[::-1, 0], 'r')
    plt.subplot(2, 2, 2)
    plt.imshow(time.cpu(), cmap=plt.get_cmap("viridis", 2**16), interpolation='nearest')

    tic = timer()
    dist, time = spline_distance_brent_(dist, time, grid, coeff)
    toc = timer() - tic
    print(f'brent (128, {steps})', toc * 1e3, 'ms')

    plt.subplot(2, 2, 3)
    plt.imshow(dist.cpu())
    plt.plot(values[:, 1], values.numpy()[::-1, 0], 'r')
    plt.subplot(2, 2, 4)
    plt.imshow(time.cpu(), cmap=plt.get_cmap("viridis", 2**16), interpolation='nearest')

    plt.title(f'[{steps}]')
    plt.show()


for steps in (8, 32, 64, 128, 256):

    tic = timer()
    dist, time = spline_distance_table(grid, coeff, steps=steps)
    toc = timer() - tic
    print(f'table ({steps})', toc * 1e3, 'ms')

    plt.subplot(2, 2, 1)
    plt.imshow(dist.cpu(), cmap=plt.get_cmap("viridis", 2**16), interpolation='nearest')
    plt.plot(values[:, 1], values.numpy()[::-1, 0], 'r')
    plt.subplot(2, 2, 2)
    plt.imshow(time.cpu(), cmap=plt.get_cmap("viridis", 2**16), interpolation='nearest')

    tic = timer()
    dist, time = spline_distance_gaussnewton_(dist, time, grid, coeff)
    toc = timer() - tic
    print(f'newton (16, {steps})', toc * 1e3, 'ms')

    plt.subplot(2, 2, 3)
    plt.imshow(dist.cpu())
    plt.plot(values[:, 1], values.numpy()[::-1, 0], 'r')
    plt.subplot(2, 2, 4)
    plt.imshow(time.cpu(), cmap=plt.get_cmap("viridis", 2**16), interpolation='nearest')

    plt.title(f'[{steps}]')
    plt.show()


foo = 0