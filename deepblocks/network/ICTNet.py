""" This is a PyTorch implementation of the ICT-Net used in """
from typing import List
from torch import nn
import torch
from deepblocks.layer import DenseBlock, SE

class ICTNet(nn.ModuleDict):
    """ ICT-Net is UNet-like network.

    ICT-Net uses DenseBlock (from DenseNet) followed by Squeeze-Excitation block in each after each downsampling/upsampling layer.
    
    Note that a ``nn.Conv2d`` is applied to input tensor before passing it Encoder, and that (DenseBlock, SE) is also used between the Encoder and Decoder.

    Note:
        For more details, check this paper: https://arxiv.org/abs/1912.09216

    Args:
        in_channels (int, Optional): Number of channels in input tensor.
                                    Default is 3.
        intermedidate_channels (int, Optional): Number of channels to produce by convolving the input tensor.
                                                Default is 32.
        out_channels (int, Optional): Number of ICT-Net output tensor. It can be seen also as the number of predicted classes.
                                        Default is 1.
        n_pool (int, Optional): Number of downsampling (and therefore upsampling) steps.
                                Default is 5.
        growth_rate (int, Optional): Parameter supplied to ``DenseBlock``. It's the number of output channels in each layer in DenseBlock.
                                    Default is 12.
        n_layers_per_block (List[int], Optional): | Number of layers to use in each ``DenseBlock``. 
                                                    ``n_layers_per_block[0]`` is layers number in the first DenseBlock of Encoder,
                                                    ``n_layers_per_block[1]`` is layers number in the second DenseBlock of Encoder, 
                                                    ...
                                                    ``n_layers_per_block[n_pool]`` is layers number in the DenseBlock between the Encoder and Decoder,
                                                    ``n_layers_per_block[n_pool+1]`` is layers number in the first DenseBlock of Decoder,
                                                    ...
                                                    ``n_layers_per_block[2*n_pool+1]`` is the layers number in the last DenseBlock of Decoder.

                                                    Note that the length of ``n_layers_per_block`` should be equal to ``2*n_pool+1``.
                                                    Default is 4 layers in every DenseBlock.
                                                    
        dropout (float, Optional): Probability of dropping a tensor entry after ``DenseBlock`` and after downsampling.
                                    Default is 2e-1.

    Shape:
        - input (torch.Tensor): A [batch, ``in_channels``, height, width] tensor. Note that `height` and `width` should be divisable by ``2**n_pools``.
        - output (torch.Tensor): A [batch, ``out_channels``, height, width] tensor.

    Example:
        >>> from deepblocks.network import ICTNet
        >>> ictnet = ICTNet(3, out_channels=33, n_pool=2, n_layers_per_block=[3]*5)
        >>> x = torch.rand(32, 3, 64, 64)
        >>> y = ictnet(x)   # y.shape = (32, 33, 64, 64)                                                    
    """

    def __init__(self, in_channels:int=3, intermediate_channels:int=32, out_channels:int=1, n_pool:int=5, growth_rate:int=12, n_layers_per_block:List[int]=[4]*11, dropout:float=0.2):
        super().__init__()

        if (n_pool*2 + 1) != len(n_layers_per_block):
            raise ValueError('The length of ``n_layers_per_block`` is not consistent with the value of ``n_pool``')

        self.n_pool = n_pool

        # first convolution
        self.add_module('conv1', nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1) )
        
        # Downsampling path
        outs = [intermediate_channels]
        for i in range(n_pool):
            self.add_module(f'down_dense_{i}', DenseBlock(outs[-1], growth_rate, n_layers_per_block[i]))
            outs.append(outs[-1] + growth_rate*n_layers_per_block[i])
            self.add_module(f'down_se_{i}', SE(outs[-1], 1))
            self.add_module(f'down_transition_{i}', self._downsample_layer(outs[-1], dropout))

        # Bottom path
        self.add_module('bottom_dense', DenseBlock(outs[-1], growth_rate, n_layers_per_block[n_pool]))
        self.add_module('bottom_se', SE(growth_rate*n_layers_per_block[n_pool], 1))

        #
        del outs[0]
        outs = outs[::-1]

        # Upsampling path
        ins = [growth_rate*n_layers_per_block[n_pool]]
        for i in range(n_pool):
            self.add_module(f'up_transition_{i}', self._upsample_layer(ins[-1]))
            self.add_module(f'up_dense_{i}', DenseBlock(outs[i]+ins[-1], growth_rate, n_layers_per_block[i+n_pool+1], dropout=dropout))
            ins.append(growth_rate*n_layers_per_block[i + n_pool+1])
            self.add_module(f'up_se_{i}', SE(ins[-1], 1))

        self.add_module('conv2', nn.Conv2d(outs[-1]+ins[-1]+ins[-2], out_channels, 1))

    def forward(self, x):
        """ """
        *_, h, w = x.shape
        pow = 2**self.n_pool
        if h%pow!=0 or w%pow!=0:
            raise ValueError(f'Input shape {(h, w)} should be divisable by {pow}')

        x = self['conv1'](x)
        xx = []
        # downsample
        for i in range(self.n_pool):
            x = self[f'down_dense_{i}'](x)
            x = self[f'down_se_{i}'](x)
            xx.append(x)
            x = self[f'down_transition_{i}'](x)
        
        x = self['bottom_se'](self['bottom_dense'](x)[:, x.size(1):, ...])  # remove input from denseblock's output
        xx = xx[::-1]

        # upsample
        for i in range(self.n_pool):
            x = self[f'up_transition_{i}'](x, output_size=xx[i].shape)
            x = torch.cat((xx[i], x), dim=1)
            n_channels = x.size(1)
            x = self[f'up_dense_{i}'](x)
            # in last iteration keep denseblock output as it is
            if i != self.n_pool-1:
                x = x[:, n_channels:, ...]  # remove input from denseblock's output
                x = self[f'up_se_{i}'](x)    

        x = self['conv2'](x)
        
        return x



    def _downsample_layer(self, channels:int, dropout:float):
        return nn.Sequential(nn.BatchNorm2d(channels),
                            nn.ReLU(), 
                            nn.Conv2d(channels, channels, 1, 1, 0), 
                            nn.Dropout(dropout), 
                            nn.MaxPool2d(2, 2))

    def _upsample_layer(self, channels:int):
        return nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1)