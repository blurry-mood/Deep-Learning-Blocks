from torch import nn


class ConvMixerLayer(nn.Module):
    """ ConvMixerLayer is the elementary "layer" that's repeated several times in ConvMixer architecture.

    This block is a sequence of **Depth-wise convolution** with residual connection followed by a **Point-wise convolution**,
    each using a ``GELU`` and ``BatchNorm``.

    Note:
        ConvMixer architecture is depicted in this paper: https://openreview.net/pdf?id=TVHS5Y4dNvM

    Args:
        in_channels (int): Number of channels in the input tensor. It's also the number of channels produced by this layer.
        kernel_size (int): Kernel size used by the depth-wise convolution. It should be less than tensor spatial dimensions.                                    

    Shape:
        - input (torch.Tensor): A [batch, ``in_channels``, height, width] tensor.
        - output (torch.Tensor): A [batch, ``in_channels``, height, width] tensor.

    Example:
        >>> from deepblocks.layer import ConvMixerLayer
        >>> mixer = ConvMixerLayer(24, kernel_size=9)
        >>> x = torch.rand(128, 24, 256, 512)
        >>> y = mixer(x)    # y.shape = (128, 24, 256, 512)
    
    """

    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      padding='same', groups=in_channels),
            nn.GELU(),
            nn.BatchNorm2d(in_channels)
        )
        self.ptwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        y = self.depthwise(x)
        x = self.ptwise(x + y)
        return x