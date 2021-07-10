'''
SE stands for Squeeze-Excitation block.
Given an input feature map F, this block independently scales each channel with a scalar computed from the F.
'''

from torch import nn
import torch


class SE(nn.Module):

    def __init__(self, in_channels: int, ratio: int = 16):
        '''
        Args:
            - in_channels: The number of channels of the input tensor.
            - ratio: an int that controls the capacity and computational cost of SE block. 
                    Bigger values reduce the two former quantities.
        '''
        super().__init__()

        assert in_channels >= ratio, f'{in_channels} should be greater than (or equal to) {ratio}.'

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(-3, -1),
            nn.Linear(in_channels, in_channels//ratio),
            nn.ReLU(),
            nn.Linear(in_channels//ratio, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ''' Computes a channels-scaled feature map.
        Args:
            - x: an input tensor whose shape: [batch, in_channels, H, W]

        Returns:
            - A tensor whose shape is [batch, in_channels, H, W]
        '''
        scalars = self.se(x)
        x = scalars.unsqueeze(-1).unsqueeze(-1) * x
        return x