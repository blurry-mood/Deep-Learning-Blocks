from typing import Tuple, Union
import torch
from torch import nn


class FReLU(nn.Module):
    """ Initializes an FReLU layer.
        
        FReLU takes a feature map, computes a depthwise convolution operation, 
        then returns the maximum value of the two feature maps in an element-wise fashion.

        Note:
            For more details, check the original paper: https://arxiv.org/abs/2007.11824
        Note:
            This implementation leaves the choice of the parametric pooling window size up to the user, it's specified in `kernel_size`.
            
        Args:
            in_channels (int): The number of channels of the input tensor.
            kernel_size (int or tuple, Optional): It could be an int or a tuple of two ints. 
                                                    Default is 1.
        
        Shape:
            - x (torch.Tensor): A [batch, channels, height, width] tensor.

        Example:
            >>> from deepblock.activation import FReLU
            >>> frelu = FReLU(8)
            >>> x = torch.rand(10, 8, 224, 224)
            >>> y = frelu(x)    # y.shape = (10, 8, 224, 224)
    """
    def __init__(self, in_channels: int, kernel_size: Union[int, Tuple[int, int]]=3):
        super().__init__()

        if torch.tensor(kernel_size).prod()%2==0:
            raise ValueError('The kernel size must be even')

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

    def forward(self, x:torch.Tensor):
        """ """
        y = self.conv(x)
        return torch.maximum(x, y)

