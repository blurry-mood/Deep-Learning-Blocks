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
from math import sqrt

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
        
        output_dim = (input_dim//num_heads)*num_heads
        self.num_heads = num_heads
        self.div = sqrt(output_dim // num_heads)

        self.query = nn.Linear(input_dim, output_dim, bias=False)
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)

        self.proj = nn.Linear(output_dim, input_dim)

    def forward(self, x:torch.Tensor):
        """ """
        B, N, D = x.shape
        query = self.query(x).reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        key = self.query(x).reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        value = self.query(x).reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        # query.shape = (batch, heads, seq_len, hidden_dim)

        weights = (query @ key.transpose(-2, -1)) / self.div
        weights = weights.softmax(dim=-1) # weights.shape = (batch, heads, hidden_dim, hidden_dim)

        x = (weights @ value).transpose(1, 2).reshape(B, N, (D // self.num_heads)* self.num_heads) # x.shape = (batch, seq_len, input_dim)
        x = self.proj(x)    # mix head outputs
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

    def __init__(self, input_dim:int, num_heads:int=8):
        super().__init__()
        
        self.num_heads = num_heads
        self.div = sqrt(input_dim)

        self.query = nn.Linear(input_dim, input_dim*num_heads, bias=False)
        self.value = nn.Linear(input_dim, input_dim*num_heads, bias=False)

        self.proj = nn.Linear(input_dim*num_heads, input_dim)

    def forward(self, x:torch.Tensor):
        """ """
        B, N, D = x.shape
        query = self.query(x).reshape(B, N, self.num_heads, D).permute(0, 2, 1, 3)
        value = self.query(x).reshape(B, N, self.num_heads, D).permute(0, 2, 1, 3)
        # query.shape = (batch, heads, seq_len, D)

        weights = (query @ x.transpose(-2, -1).unsqueeze(1)) / self.div
        weights = weights.softmax(dim=-1) # weights.shape = (batch, heads, hidden_dim, hidden_dim)

        x = (weights @ value).transpose(1, 2).reshape(B, N, D) # x.shape = (batch, seq_len, input_dim)
        x = self.proj(x)    # mix head outputs
        return x