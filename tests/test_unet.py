from deepblocks.network import UNet
import torch

def test_shape():
    unet = UNet()
    x = torch.rand(3, 1, 256, 256)
    y = unet(x) 
    assert y.shape == (3, 2, 256, 256)