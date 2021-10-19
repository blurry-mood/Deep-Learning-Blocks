""" This is a PyTorch implementation of the ICT-Net"""

from torch import nn
from ..layer import ConvMixerLayer


class ConvMixer(nn.Module):
    """ Initialize a ConvMixer module for image classification/regression.  
    
    ConvMixer consists of a patch embedding layer followed by repeated applications 
    of a simple fully-convolutional block.

    Note:
        ConvMixer architecture is depicted in this paper: https://openreview.net/pdf?id=TVHS5Y4dNvM

    Args:
        in_channels (int, Optional): Number of channels in the input tensor.
                                    Default is 3.        
        num_classes (int, Optional): The number of outputs per image.
                                        Default is 1000.
        patch_dim (int, Optional): Patch embeddings spatial size.
                                        Default is 10.
        hidden_dimension (int, Optional): Patch embeddings dimenion.
                                    Default is 7.
        kernel_size (int, Optional): The kernel size of the depthwise convolutional layer.
                                    Default is 9.

    Shape:
        - input (torch.Tensor): A [batch, ``in_channels``, height, width] tensor.
        - output (torch.Tensor): A [batch, ``num_classes``] tensor.

    Example:
        >>> from deepblocks.network import ConvMixer
        >>> mixer = ConvMixer(in_channels=24, num_classes=100)
        >>> x = torch.rand(128, 24, 256, 512)
        >>> y = mixer(x)    # y.shape = (128, 100)
    
    """


    def __init__(self, in_channels=3, num_classes=1000, hidden_dimension=10, depth=7, patch_dim=7, kernel_size=9):
        super().__init__()

        self.patch_embed = nn.Sequential(nn.Conv2d(in_channels, hidden_dimension, kernel_size=patch_dim, stride=patch_dim),
                                         nn.GELU(),
                                         nn.BatchNorm2d(hidden_dimension))
        
        self.hidden = nn.Sequential(*[ConvMixerLayer(hidden_dimension, kernel_size) for _ in range(depth)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_dimension, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.hidden(x)
        x = self.linear(self.flatten(self.avgpool(x)))
        return x