'''
This file contains the implementation of the Funnel ReLU activation function.
The layerReLU F takes a feature map, computes a convolution operation, 
    then return the maximum value of the two feature maps in an element-wise fashion.
'''

from typing import Tuple, Union
import torch
from torch import nn


class FReLU(nn.Module):

    def __init__(self, in_channels: int, kernel_size: Union[int, Tuple[int, int]]=3):
        '''Initializes the layer: a conv2d layer with groups==in_channels.
            The padding of the convolution is computed such as the resulting feature map
            has the same shape as the input.   
            The convolution has no bias term.
        Args:
            - in_channels: The number of channels of the input tensor.
            - kernel_size: could be an int or a tuple of two ints.
        '''
        super().__init__()

        assert torch.tensor(kernel_size).prod()%2==1, 'The kernel size is even'

        # Pad the image with zeros as much as needed to
        # preseve the width and height of the input tensor
        if isinstance(kernel_size, int):
            padding = kernel_size//2
        elif len(kernel_size) == 1:
            kernel_size = kernel_size[0]
            padding = kernel_size//2
        else:
            padding = (kernel_size[0]//2, kernel_size[1]//2)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, bias=False,
                              kernel_size=kernel_size, stride=1, padding=padding, groups=in_channels)

    def forward(self, x):
        ''' Computes the funnel relu activation.
        Args:
            - x: an input tensor whose shape: [batch, in_channels, H, W]

        Returns:
            - A tensor whose shape is [batch, in_channels, H, W]
        '''
        y = self.conv(x)
        return torch.maximum(x, y)

