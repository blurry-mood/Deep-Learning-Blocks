from typing import List
from torch import nn
import torch

class UNet(nn.Module):
    """ Initialize a UNet module for semantic segmentation.

    UNet constists mainly of two paths: Encoder & Decoder.
    The Encoder is a sequence of convolutions and max pooling layers that double the number of channels 
    and cut in half both width and height of feature maps.
    In contrast, the Decoder is a sequence of transpose convolutions and convolution that double both 
    width & height of feature maps but cutting in half number of channels.

    This implementation introduces more parameters to make the architecture flexible.
    Other than ``in_channels`` & ``out_channels``, the depth of Encoder (and Decoder) is variable, 
    as well as the possibility of using batch normalization & dropout after every convolution layer in the network.

    Note:
        The architecture is described in this paper:https://arxiv.org/abs/1505.04597

    Note:
        To use the vanilla UNet as depicted in the paper, initialize an instance without passing any arguments. 
        Otherwise, you customize the network by changing default values.
        
    Note:    
        This implementation uses ``Conv2d`` with a padding of 1 to make sure input & output share same spatial dimensions.
    
    Args:
        n_pool (int, Optional): Depth of Encoder. Number of times to apply max pooling to input tensor throughtout the encoding phase.
                                Default is 4.
        in_channels (int, Optional): Number of channels in input tensor.
                                    Default is 1.
        out_channels (int, Optional): Number of channels in output tensor.
                                    Default is 2.
        batchnorm (bool, Optional): Specifies whether or not to batch normalize feature maps produced by every ``nn.Conv2d`` layer.
                                    Default is false.
        dropout (float, Optional): Probability of dropping an entry in feature maps produced by ``nn.Conv2d`` (or ``nn.BatchNorm2d`` if batchnorm is set to true).
                                    Default is 0.

    Shape:
        - input (torch.Tensor): A [batch, ``in_channels``, height, width] tensor. 
            The `height` and `width` must be larger enough such as the last tensor after Encode must be larger than (h=3, w=3).
        - output (torch.Tensor): A [batch, ``out_channels``, height, width] tensor. 

    Example:
        >>> from deepblocks.network import UNet
        >>> unet = UNet()
        >>> x = torch.rand(3, 1, 572, 572)
        >>> y = unet(x) # y.shape = (3, 2, 572, 572)
    """

    def __init__(self, n_pool:int=4, in_channels:int=1, out_channels:int=2, batchnorm:bool=False, dropout:float=0):
        super().__init__()

        self.conv1 = nn.Sequential(
            *self._make_conv(in_channels, 64, batchnorm=batchnorm, dropout=dropout),
            *self._make_conv(64, 64, batchnorm=batchnorm, dropout=dropout),
        )

        in_channels:List = [64]

        self.encoder = nn.ModuleList()
        for _ in range(n_pool):
            in_channels.append(in_channels[-1]*2)
            self.encoder.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                *self._make_conv(in_channels=in_channels[-2], out_channels=in_channels[-1], batchnorm=batchnorm, dropout=dropout),
                *self._make_conv(in_channels=in_channels[-1], out_channels=in_channels[-1], batchnorm=batchnorm, dropout=dropout)  
            ))

        self.upsampler = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(n_pool):
            self.upsampler.append(nn.ConvTranspose2d(in_channels=in_channels[-1], out_channels=in_channels[-2-i], kernel_size=2, stride=2),)
            self.decoder.append(nn.Sequential(
                *self._make_conv(in_channels=2*in_channels[-i-2], out_channels=in_channels[-i-2], batchnorm=batchnorm, dropout=dropout),
                *self._make_conv(in_channels=in_channels[-i-2], out_channels=in_channels[-i-2], batchnorm=batchnorm, dropout=dropout),
            ))
            in_channels[-1] = in_channels[-2-i]

        self.output = nn.Sequential(*self._make_conv(in_channels=in_channels[-1], out_channels=out_channels, batchnorm=batchnorm, dropout=dropout))

    def forward(self, x:torch.Tensor):
        """ """
        xx = [self.conv1(x)] # store encoder outputs
        for i in range(len(self.encoder)):
            xx.append(self.encoder[i](xx[-1]))
        
        xx = xx[::-1] # last output becomes first
        x = xx[0]

        for i in range(len(self.decoder)):
            x = self.upsampler[i](x, xx[i+1].shape )
            x = torch.cat((xx[i+1], x), dim=1)    # concatenate along channels dimensions
            x = self.decoder[i](x)

        return self.output(x)


    def _make_conv(self, in_channels:int, out_channels:int, batchnorm:bool, dropout:float):
        mods = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)]
        if batchnorm:
            mods.append(nn.BatchNorm2d(num_features=out_channels))
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
        mods.append(nn.ReLU())
        return mods