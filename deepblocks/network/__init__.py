""" **Network** contains implmentation of various model architecture ready to be used out of the box.

The implementation of networks offer more parameters (than in original work) to customize the final architecture.

It contains:
    - U-Net
    - ICT-Net 
    - ConvMixer
"""

from .UNet import UNet
from .ConvMixer import ConvMixer
from .ICTNet import ICTNet