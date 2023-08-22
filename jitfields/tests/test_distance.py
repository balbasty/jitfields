import cppyy
cppyy.set_debug(True)

from jitfields.distance import (
    l1_distance_transform,
    euclidean_distance_transform
)
from .utils import get_test_devices, init_device
import torch
import pytest

devices = get_test_devices()


@pytest.mark.parametrize("device", devices)
def test_l1(device):
    device = init_device(device)

    mask = [[1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]]

    dist = [[4, 3, 2, 2, 3, 4],
            [3, 2, 1, 1, 2, 3],
            [2, 1, 0, 0, 1, 2],
            [2, 1, 0, 0, 1, 2],
            [3, 2, 1, 1, 2, 3],
            [4, 3, 2, 2, 3, 4]]

    mask = torch.as_tensor(mask, device=device)
    dist = torch.as_tensor(dist, device=device)
    pred = l1_distance_transform(mask)
    assert torch.allclose(dist.to(pred), pred)


@pytest.mark.parametrize("device", devices)
def test_l2(device):
    device = init_device(device)

    mask = [[1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]]

    dist = [[8, 5, 4, 4, 5, 8],
            [5, 2, 1, 1, 2, 5],
            [4, 1, 0, 0, 1, 4],
            [4, 1, 0, 0, 1, 4],
            [5, 2, 1, 1, 2, 5],
            [8, 5, 4, 4, 5, 8]]

    mask = torch.as_tensor(mask, device=device)
    dist = torch.as_tensor(dist, device=device)
    pred = euclidean_distance_transform(mask).square()
    assert torch.allclose(dist.to(pred), pred)
