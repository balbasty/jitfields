import torch
from jitfields import identity_grid
import math


def circle(shape, width=0.6, **backend):
    """Generate an image with a circle"""
    backend.setdefault('device', 'cpu')
    backend.setdefault('dtype', torch.get_default_dtype())
    width *= min(shape)

    mask = identity_grid(shape, device=backend['device'])
    mask -= (torch.as_tensor(shape, **backend) - 1) / 2
    mask = mask.square().sum(-1).sqrt()
    mask = mask < width/2
    return mask.to(backend['dtype'])


def square(shape, width=0.6, **backend):
    """Generate an image with a square"""
    backend.setdefault('device', 'cpu')
    backend.setdefault('dtype', torch.get_default_dtype())

    start = [s - int(math.floor(width * s))//2 for s in shape]
    stop = [s + s0 - int(math.floor(width * s)) for s, s0 in zip(shape, start)]
    slicer = [slice(s0 or None, (-s1) or None) for s0, s1 in zip(start, stop)]

    mask = torch.zeros(shape, **backend)
    mask[tuple(slicer)] = 1
    return mask
