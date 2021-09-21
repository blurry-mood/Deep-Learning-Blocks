r""" **Layer** implements useful PyTorch layers that can be easily incorporated within every model.

It contains:
    - SE (Squeeze-Excitation) for Convnets,
    - FlipConv2d (Special Conv2d layer that's almost invariant to image flipping) for Convnets,
    - MultiHeadAttention implemented in the transformers paper (Attention is all you need).
    - MultiHeadAttentionV2 is a modified version.
"""
from .FlipConv2d import FlipConv2d
from .SE import SE
from .MultiHeadAttention import MultiHeadAttention, MultiHeadAttentionV2