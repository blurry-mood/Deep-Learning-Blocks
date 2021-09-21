from torch import nn

class SE(nn.Module):
    """ SE stands for Squeeze-Excitation. 

    Given an input feature map F, this block independently scales each channel with a scalar, which lies between 0 and 1, computed from the F.  

    SE returns a new scaled feature map by using the Inception module method not the ResNet's.

    Note:
        For more insights, check the original paper: https://arxiv.org/abs/1709.01507

    Args:
        in_channels (int): The number of channels of the input tensor.
        ratio (int, Optional): It controls the capacity and computational cost of the SE block. 
                                Bigger values reduce the two aforementioned quantities. 
                                Note that `ratio` should positive and less than or equals to `in_channels`.
                                Default is 16.

    Shape:
        - x (torch.Tensor): A [batch, in_channels, H, W] tensor

    Example:
        >>> from deepblocks.layer import SE
        >>> se = SE(16, 4)
        >>> x = torch.rand(10, 16, 32, 32)
        >>> y = se(x)   # y.shape = (10, 16, 32, 32)
    """

    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()

        if not (in_channels >= ratio>0):
            raise ValueError(f'{in_channels} should positive and be greater than (or equal to) {ratio}')

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(-3, -1),
            nn.Linear(in_channels, in_channels//ratio),
            nn.ReLU(),
            nn.Linear(in_channels//ratio, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """  """
        scalars = self.se(x)
        x = scalars.unsqueeze(-1).unsqueeze(-1) * x
        return x