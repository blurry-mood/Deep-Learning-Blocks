from torch import nn
import torch


class DenseBlock(nn.ModuleDict):
    """ DenseBlock is the elementary "layer" that's repeated several times in DenseNet architecture.

    This block is a sequence of (``nn.BatchNorm2d`` + ``nn.ReLU`` + ``nn.Conv2d`` + ``nn.Dropout``) layers, where each one takes as input all previous feature maps 
    produced by the previous layer. Those feature maps are concatenated along the channel dimension.

    The output tensor of this block has ``input_channels`` + ``growth_rate`` * ``num_layers`` channels, 
    the rest of dimensions have the same number of elements as input.

    Note:
        DenseNet architecture is depicted in this paper: https://arxiv.org/abs/1608.06993v5

    Args:
        in_channels (int): Number of channels in the input tensor.
        growth_rate (int, Optional): Number of output channels in each layer in the block, 
                                    i.e ``output_channels`` in every conv2d is ``growth_rate``.
                                    Default is 12.
        num_layers (int, Optional): Number of consecutive [``nn.BatchNorm2d``, ``nn.ReLU``, ``nn.Conv2d``] layers. 
                                    Default is 4.
        dropout (float, Optional): Probability of the ``Dropout``.
                                    Default is 1e-1.
        activation (nn.Module, Optional): The desired activation to instead of ``nn.ReLU``. 
                                            If ``None`` is passed, the activation is skipped,
                                            otherwise, the supplied class must instantiable without passing any arguments.
                                            Default is ``nn.ReLU``
        batchnorm (nn.Module, Optional): The batch normalization layer to use. 
                                        Note that only ``nn.BatchNorm2d`` and ``None`` are the possible values.
                                        Default is ``nn.BatchNorm2d``.

    Shape:
        - x (torch.Tensor): A [batch, channels, height, width] input tensor.

    Example:
        >>> from deepblocks.layer import DenseBlock
        >>> dense = DenseBlock(24, num_layers=5, batchnorm=None)
        >>> x = torch.rand(128, 24, 256, 512)
        >>> y = dense(x)    # y.shape = (128, 84, 256, 512)
    
    """

    def __init__(self, in_channels:int, growth_rate:int=12, num_layers:int=4, dropout:float=0.1, activation:nn.Module=nn.ReLU, batchnorm:nn.Module=nn.BatchNorm2d):
        super().__init__()

        for i in range(num_layers):
            input = in_channels + i * growth_rate
            self.add_module(f'layer_{i}', self._dense_layer(input, growth_rate, activation, batchnorm, dropout))

    def forward(self, x:torch.Tensor):
        """ Returns a tensor with ``input_channels`` + ``growth_rate`` * ``num_layers`` channels. """
        for _, layer in self.items():
            x = torch.cat([x, layer(x)], dim=1) # concatenate along the channel dimension
        return x


    def _dense_layer(self, in_channels, out_channels, act, bn, dropout):
        models = []
        if bn is not None:
            models.append(bn(in_channels))
        if act is not None:
            models.append(act())
        models.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        if dropout>0:
            models.append(nn.Dropout(dropout))
        return nn.Sequential(*models)

