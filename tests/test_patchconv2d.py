from deepblocks.layer import PatchConv2d
import torch

def test_shape():
    patch = PatchConv2d(3, 6, 3, padding=2, stride=2)
    x = torch.rand(10, 3, 32, 32)
    y = patch(x)
    assert y.shape == (10, 6, 17, 17)