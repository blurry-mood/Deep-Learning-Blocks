
from typing import Union

import torch
from torch import nn
from torch.nn.functional import fold
from torch.nn.common_types import _size_2_t

class PatchConv2d(nn.Conv2d):
    """ PatchConv2d is a Conv2d-like layer that computes L2-distance between each patch in input image with the kernel.

    A typical ``Conv2d`` layer computes the output tensor as linear combination of values in each patch as follows:
    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    however, this layer, instead of linear combination, computes squared error between every patch and the kernel as follows:
    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} (\text{weight}(C_{\text{out}_j}, k) - \text{input}(N_i, k))^2

    ``PatchConv2d`` takes the same arguments as ``Conv2d``, but ``groups`` & ``padding_mode``, 
    they have a default value of ``1`` and ``zeros`` respectively.
        

    Args:
        in_channels (int): Number of channels in the input image
        kernel_size (int): Number of channels in the output image                                    
        kernel_size (int or Tuple): Size of the "convolving" kernel
        stride (int or Tuple, Optional): Stride of the "convolution"
                                        Default is 1
        dilation (int or tuple, optional): Spacing between kernel elements
                                            Default: 1
        bias (bool, Optional): If ``True``, adds a learnable bias to the output
                                Default is ``True``

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (``out_channels``, ``in_channels``*``kernel_size[0]``*``kernel_size[1]``, 1)
            The values of these weights are sampled from a standard gaussian distribution.
        bias (Tensor):   the learnable bias of the module of shape (``out_channels``, 1).
            If :attr:`bias` is ``True``, then the initail values of these weights are all equal to zero.

    Shape:
        - Input: :math:`(N, ``in_channels``, H_{in}, W_{in})`
        - Output: :math:`(N, ``out_channels``, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Example:
        >>> from deepblocks.layer import PatchConv2d
        >>> patch = PatchConv2d(3, 6, 3, padding=2, stride=2)
        >>> x = torch.rand(10, 3, 32, 32)
        >>> y = patch(x)    # y.shape = (10, 6, 17, 17)
    
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=bias, padding_mode='zeros', device=device, dtype=dtype)

        self.unfold = nn.Unfold(self.kernel_size, self.dilation, self.padding, self.stride)
        self.weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels*self.kernel_size[0]*self.kernel_size[1], 1, dtype=dtype, device=device))
        self.bias = None if not bias else nn.Parameter(torch.zeros(self.out_channels, 1, dtype=dtype, device=device))

    def forward(self, x):
        out_shape = x.shape[2:]
        out_shape = [int((out_shape[i]+2*self.padding[i]-self.dilation[i]*(self.kernel_size[i]-1)-1)/self.stride[i])+1 
                    for i in [0, 1]]
        # x.shape = (n, c, h, w)
        x = self.unfold(x)  # x.shape = (n, c*K, L)
        x = x.unsqueeze(1)  # x.shape = (n, 1, c*K, L)
        x = (self.weight - x).pow(2) # x.shape = (n, out_channels, c*K, L)
        x = x.sum(axis=2)   # x.shape = (n, out_channels, L)
        if self.bias is not None:
            x = x + self.bias
        x = fold(x, output_size=out_shape, kernel_size=(1, 1))
        return x


if __name__ == '__main__':
    patch = PatchConv2d(3, 6, 3, padding=2, stride=2)
    x = torch.rand(10, 3, 32, 32)
    y = patch(x)
    print(y.shape)