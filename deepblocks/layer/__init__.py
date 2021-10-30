r""" **Layer** implements useful PyTorch layers that can be easily incorporated within every model.

It contains:
    - FlipConv2d (Special Conv2d layer that's almost invariant to image flipping) for Convnets,
    - PatchConv2d (Special Conv2d layer that computes L2 distance between image patches and kernel) for Convnets,

    - SE (Squeeze-Excitation) for Convnets,
    - DenseBlock (implemented in DenseNet) for Convnets,
    - MultiHeadAttention implemented in the transformers paper (Attention is all you need).
    - MultiHeadAttentionV2 is a modified version.
    - ConvMixerLayer (implemented in ConvMixer) for Convnets
"""
from .FlipConv2d import FlipConv2d
from .SE import SE
from .MultiHeadAttention import MultiHeadAttention, MultiHeadAttentionV2
from .ConvMixerLayer import ConvMixerLayer
from .PatchConv2d import PatchConv2d
from .DenseBlock import DenseBlock