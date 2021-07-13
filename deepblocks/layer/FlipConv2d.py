'''
FlipConv2d is a special Conv2d layer whose result is invariant against horizontal & vertical flipping
'''

import torch
from torch import nn
from torch.nn.functional import conv2d


class FlipConv2d(nn.Module):

    def __init__(self, h_invariant: bool = True, v_invariant: bool = True, **kwargs):
        super().__init__()

        self.h_invariant = h_invariant
        self.v_invariant = v_invariant
        self.kwargs = kwargs

        # Extract necessary info
        kernel_size = kwargs.pop('kernel_size', (3, 3))
        in_channels = kwargs.pop('in_channels')
        out_channels = kwargs.pop('out_channels')
        groups = kwargs.pop('groups', 1)
        kwargs['groups'] = groups

        # Two conditions needs to be met
        assert isinstance(kernel_size, int) or isinstance(
            kernel_size, tuple), 'The kernel size must be either a tuple or an int'
        assert torch.tensor(kernel_size).prod(
        ) % 2 == 1, 'The kernel dimensions must be odd'

        # Store the kernel size in 'kernel_size' tuple
        if isinstance(kernel_size, int) or len(kernel_size) == 1:
            if not isinstance(kernel_size, int):
                kernel_size = kernel_size[0]
            kernel_size = (kernel_size, kernel_size)

        # Lower the kernel dimension along axes which it's flip-invariant
        if h_invariant:
            kernel_size = (kernel_size[0], kernel_size[1]//2+1)
        if v_invariant:
            kernel_size = (kernel_size[0]//2+1, kernel_size[1])

        # Register the model paramters: bias & kernel weights
        self.bias = nn.Parameter(3*torch.randn(out_channels))
        self.kernel = nn.Parameter(
            3*torch.randn(out_channels, in_channels//groups, kernel_size[0], kernel_size[1]))

    def forward(self, x):
        kernel = self.kernel

        # Flip the kernel vertically
        if self.v_invariant:
            v_inds = torch.arange(start=kernel.size(-2)-2, end=-1, step=-1)
            kernel = torch.cat((kernel, kernel[..., v_inds, :]), axis=-2)

        # Flip the kernel horizontally
        if self.h_invariant:
            h_inds = torch.arange(start=kernel.size(-1)-2, end=-1, step=-1)
            kernel = torch.cat((kernel, kernel[..., h_inds]), axis=-1)

        return conv2d(x, kernel, bias=self.bias, **self.kwargs)
