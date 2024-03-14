from jitfields.distance import mesh_distance_signed
import torch
from math import pi, cos, sin, sqrt
import matplotlib.pyplot as plt

def make_icosahedron():
    t = (1 + sqrt(5)) / 2
    vertices = [
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],
        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],
        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ]
    faces = [
        [ 0, 11,  5],
        [ 0,  5,  1],
        [ 0,  1,  7],
        [ 0,  7, 10],
        [ 1,  5,  9],
        [ 5, 11,  4],
        [11, 10,  2],
        [10,  7,  6],
        [ 7,  1,  8],
        [ 3,  9,  4],
        [ 3,  4,  2],
        [ 3,  2,  6],
        [ 3,  6,  8],
        [ 3,  8,  9],
        [ 4,  9,  5],
        [ 2,  4, 11],
        [ 6,  2, 10],
        [ 8,  6,  7],
        [ 9,  8,  1],
    ]
    vertices = torch.as_tensor(vertices, dtype=torch.float32)
    vertices /= sqrt(1 + t*t)  # make vertices lie on the unit sphere
    faces = torch.as_tensor(faces, dtype=torch.int32)
    return vertices, faces


def plane_intersect(vertices, faces, norm, point):
    """Compute the 2d mesh of the intersection of a 3d mesh witha plane

    Parameters
    ----------
    vertices : (N, 3) tensor
    faces : (M, 3) tensor
    norm : (3) tensor
    point : (3) tensor

    Returns
    -------
    vertices : (N', 2) tensor
    faces : (M', 2) tensor

    """
    

# convex shape

vertices, faces = make_icosahedron()
coord = torch.stack(torch.meshgrid(torch.linspace(-2, 2, 128), 
                                   torch.linspace(-2, 2, 128), 
                                   torch.linspace(-2, 2, 128)), -1)

sdt = mesh_distance_signed(coord, vertices, faces)

plt.imshow(sdt[len(sdt)//2], vmin=-1.5, vmax=1.5, cmap='coolwarm', interpolation='nearest')
plt.show(block=False)

foo = 0