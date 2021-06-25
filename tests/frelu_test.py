''' Test the initialization and the forward pass of the FReLU layer
'''

from deepblocks.CNN.FReLU import FReLU
import torch


def test_init():
    in_channels, kernel_size = torch.randint(low=1, high=50, size=(2,)).tolist()
    frelu = FReLU(in_channels, kernel_size)
    
    # Check the number of parameters
    assert sum(p.numel() for p in frelu.conv.parameters() if p.requires_grad) == kernel_size**2*in_channels

def test_forward():
    batch, width, height, in_channels, kernel_size = torch.randint(low=1, high=50, size=(5,)).tolist()
    # Taking for granted that the kernel_size is odd
    frelu = FReLU(in_channels, 2*kernel_size+1)
    
    img = torch.rand(batch, in_channels, height, width)
    # Make sure that no matter what the height and width, frelu would still work
    y = frelu(img)
    assert tuple(y.shape)==(batch, in_channels, height, width)