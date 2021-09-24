from deepblocks.network import ICTNet
import torch
from numpy.random import randint

def test_output_shape():
    for in_channels in randint(low=1, high=20, size=(2,)):
        for out_channels in randint(low=1, high=20, size=(2,)):
            for intermediate_channels in randint(low=10, high=50, size=(2,)):
                for height in [64, 128, 96]:
                    for width in [64, 32, 96]:
                        x = torch.rand(2, in_channels, height, width)
                        ictnet = ICTNet(in_channels, intermediate_channels, out_channels, growth_rate=3)
                        y = ictnet(x)
                        assert y.shape == (2, out_channels, height, width)
            