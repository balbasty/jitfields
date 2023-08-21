from jitfields.distance import mesh_distance_signed, mesh_distance
import torch
from math import pi, cos, sin
import matplotlib.pyplot as plt

# convex shape
if False:

    n = 8
    vertices = [-i*360/n for i in range(n)]
    vertices = map(lambda x: x*pi/180, vertices)
    vertices = map(lambda x: [cos(x), sin(x)], vertices)
    vertices = list(vertices)

    edges = [[i, i+1] for i in range(n)]
    edges[-1][-1] = 0

    vertices = torch.as_tensor(vertices, dtype=torch.float32)
    edges = torch.as_tensor(edges, dtype=torch.int32)
    coord = torch.stack(torch.meshgrid(torch.linspace(-2, 2, 128), 
                                    torch.linspace(-2, 2, 128)), -1)

    sdt = mesh_distance_signed(coord, vertices, edges)

    v = torch.cat([vertices, vertices[:1]], 0)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(sdt, vmin=-1.5, vmax=1.5, cmap='coolwarm', interpolation='nearest')
    plt.plot(*(v.T.flip(0) * 32 + 64), 'k')
    plt.subplot(1, 2, 2)
    plt.imshow(sdt < 0, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
    plt.plot(*(v.T.flip(0) * 32 + 64), 'r')
    plt.show(block=False)

    foo = 0

# nonconvex shape
if True:

    n = 8
    vertices = [-i*360/n for i in range(n)]
    vertices = map(lambda x: x*pi/180, vertices)
    vertices = map(lambda x: [cos(x), sin(x)], vertices)
    vertices = [[0.5*c, 0.5*s] if i%2 else [c, s] for i, (c, s) in enumerate(vertices)]
    vertices = list(vertices)

    edges = [[i, i+1] for i in range(n)]
    edges[-1][-1] = 0

    vertices = torch.as_tensor(vertices, dtype=torch.float32)
    edges = torch.as_tensor(edges, dtype=torch.int32)
    coord = torch.stack(torch.meshgrid(torch.linspace(-2, 2, 128), 
                                    torch.linspace(-2, 2, 128)), -1)

    sdt = mesh_distance_signed(coord, vertices, edges)

    v = torch.cat([vertices, vertices[:1]], 0)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(sdt, vmin=-1.5, vmax=1.5, cmap='coolwarm', interpolation='nearest')
    plt.plot(*(v.T.flip(0) * 32 + 64), 'k')
    plt.subplot(1, 2, 2)
    plt.imshow(sdt < 0, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
    plt.plot(*(v.T.flip(0) * 32 + 64), 'r')
    plt.show(block=False)

    foo = 0