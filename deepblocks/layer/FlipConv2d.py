import torch
from torch import nn
from torch.nn.functional import conv2d as _conv2d


class FlipConv2d(nn.Module):
    """ FlipConv2d is special case of the standard nn.Conv2d. They differs in the kernal structure.
    
    This kernel structure enables making the layer invariant to flipping. 
    Formally, if the feature map X' is the horizonatally-flipped version of X 
    and `flipconv2d` is this layer when `h_invariant` is true, then
    flipconv2d(X`) equals the horizonatally-flipped version of flipconv2d(X).
    This is achieved by forcing the kernel plane (indexed by the last two dimensions) to be symmetric with respect to its vertical axis.

    This module accepts the same parameters as nn.Conv2d, as well as `h_invariant` & `v_invariant`.

    if `h_invariant` and `v_invariant` are both false, this layer acts like nn.Conv2d.

    Args:
        h_invariant (bool, Optional): If true, the kernels are symmetric with respect to vertical axis. Default: True.
        v_invariant (bool, Optional): If true, the kernels are symmetric with respect to horizontal axis. Default: True.
        **kwargs: Arguments passed for the standard nn.Conv2d layer.   
                    Note that `in_channels`, `out_channels` & `kernel_size` are required. The others are optional and take the default value of nn.Conv2d.

    Shape:
        - x (torch.Tensor): A [B, C, H, W] input feature map.

    Attributes:
        kernel (torch.Tensor): The kernel tensor used for the cross-correlation operation.

    Example:
        >>> from deepblocks.layer import FlipConv2d
        >>> conv = FlipConv2d(v_invariant=False, in_channels=2, out_channels=4, kernel_size=5, stride=2, bias=False)
        >>> x = torch.rand(10, 2, 32, 32)
        >>> fm = conv(x)

    """
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
        self.bias = None if not kwargs.pop('bias', False) else nn.Parameter(torch.randn(out_channels))
        self._kernel = nn.Parameter(
            torch.randn(out_channels, in_channels//groups, kernel_size[0], kernel_size[1]))

    @property
    def kernel(self):
        """ Returns kernel weights used for the Conv2d operation.
        """
        kernel = self._kernel

        # Flip the kernel vertically
        if self.v_invariant:
            v_inds = torch.arange(start=kernel.size(-2)-2, end=-1, step=-1)
            kernel = torch.cat((kernel, kernel[..., v_inds, :]), axis=-2)

        # Flip the kernel horizontally
        if self.h_invariant:
            h_inds = torch.arange(start=kernel.size(-1)-2, end=-1, step=-1)
            kernel = torch.cat((kernel, kernel[..., h_inds]), axis=-1)

        return kernel

    def forward(self, x:torch.Tensor):
        """ """
        return _conv2d(x, self.kernel, bias=self.bias, **self.kwargs)