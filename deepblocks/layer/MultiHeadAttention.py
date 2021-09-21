r""" This module contains the implementaion of the multi-head attention mechanism from the Transformers paper 2017, 
as well as a slightly modified version.

Namely:
    - MultiHeadAttention,
    - MultiHeadAttentionV2: 
        This is an implementation of the MHA from  but with slight changes.   
        1. The hidden dimension is always the same as the input,  
        2. The key is the input itself.
"""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """ Initializes a Multi-Head Attention module.

    Note:
        - Check the original paper for more insights: https://arxiv.org/abs/1706.03762

    Args:
        input_dim (int): denotes the number of features of inputs in the sequence.
        num_heads (int, Optional): denotes the number of heads to use. Default is 8.

    Shape:
        - x (torch.Tensor): A tensor with shape [batch_size, sequence_length, input_dim].

    Example:
        >>> from deepblocks.layer import MultiHeadAttention
        >>> mha = MultiHeadAttention(22, 8)
        >>> x = torch.rand(10, 44, 22)
        >>> y = mha(x)  # y.shape = (10, 44, 22)
    """

    def __init__(self, input_dim:int, num_heads:int=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = 1/head_dim**0.5

        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=False)
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, x:torch.Tensor):
        """ """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MultiHeadAttentionV2(nn.Module):
    """ Initializes a modified Multi-Head Attention module, labelled V2.

    It has two differences with the vanilla version, namely, the hidden dimension in each head is a constant equalling to `input_dim`,
    and the projection matrix used to obtain the key is always the identity matrix, to reduce the redundancy of parameters 
    engendered by the multiplication of two learnable weight matrices: Key & Query.

    This verion is more efficient than the vanilla when the number of heads equals 1, while still having the same capacity.

    Args:
        input_dim (int): denotes the number of features of inputs in the sequence.
        num_heads(int, Optional): denotes the number of heads to use. Default is 1.

    Shape:
        - x (torch.Tensor): A tensor with shape [batch_size, sequence_length, input_dim].

    Example:
        >>> from deepblocks.layer import MultiHeadAttentionV2
        >>> mha2 = MultiHeadAttentionV2(22, 8)
        >>> x = torch.rand(10, 44, 22)
        >>> y = mha2(x)  # y.shape = (10, 44, 22)

    """

    def __init__(self, input_dim: int, num_heads: int=1):
        super().__init__()

        self.num_heads = num_heads
        self.input_dim = input_dim

        # the key projection is the identity, thus ommited
        self.qv = nn.Linear(input_dim, input_dim*2*num_heads, bias=False)
        self.concat = nn.Linear(input_dim*num_heads, input_dim)

    def forward(self, x:torch.Tensor):
        """ """
        qv = self.qv(x).unflatten(-1, (self.num_heads,
                                       self.input_dim, 2)).permute(4, 0, 2, 1, 3)
        q, v = qv[0], qv[1]
        alpha = torch.matmul(q, x.transpose(-1, -2).unsqueeze(1))/self.input_dim**.5
        alpha = torch.softmax(alpha, dim=-1)

        # compute new x
        x = torch.matmul(alpha, v).transpose(1, 2).flatten(-2)
        
        # concatenate from heads
        x = self.concat(x)

        return x