from deepblocks.layer import DenseBlock
import torch

def test_output_shape():
    for in_channels in range(1, 15, 3):
        for growth_rate in range(5, 20, 3):
            for num_layers in range(1, 5):
                dense = DenseBlock(in_channels, growth_rate, num_layers)
                x = torch.rand(2, in_channels, 64, 65)
                y = dense(x)
                assert y.shape == (2, in_channels+growth_rate*num_layers, 64, 65)