from deepblocks.network import ConvMixer
import torch

def test_shape():
    convmixer = ConvMixer()
    x = torch.rand(3, 3, 64, 64)
    y = convmixer(x) 
    assert y.shape == (3, 1000)