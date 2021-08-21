""" This is an implementation of the MHA from the Transformers paper 2017 but with slight changes.
1. The hidden dimension is always the same as the input
2. The key is the input itself
"""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """ Initializes a Multi-Head Attention module.

    Args:
        - input_dim(Integer): denotes the number of features in the input sequence.
        - num_heads(Integer): denotes the number of heads to use. 
                                Note that the hidden dimension is totally independant of the number of heads,
                             it's always equal to input_dim.
        - drop(Float): refers to the probability used by the dropout layer in cancelling the weights in the weighted sum. 
                        It should be in interval [0, 1].

    Forward args:
        - x(Tensor): A float tensor with shape [batch_size, sequence_length, input_dim].
    """

    def __init__(self, input_dim: int, num_heads: int, drop:float ):
        super().__init__()

        self.num_heads = num_heads
        self.input_dim = input_dim

        # the key projection is the identity, thus ommited
        self.qv = nn.Linear(input_dim, input_dim*2*num_heads)
        self.drop = nn.Dropout(drop)
        self.concat = nn.Linear(input_dim*num_heads, input_dim)

    def forward(self, x: torch.Tensor):
        qv = self.qv(x).unflatten(-1, (self.num_heads,
                                       self.input_dim, 2)).permute(4, 0, 2, 1, 3)
        q, v = qv[0], qv[1]
        alpha = torch.matmul(q, x.transpose(-1, -2).unsqueeze(1))/self.input_dim**.5
        alpha = torch.softmax(alpha, dim=-1)
        alpha = self.drop(alpha)

        # compute new x
        x = torch.matmul(alpha, v).transpose(1, 2).flatten(-2)
        
        # concatenate from heads
        x = self.concat(x)

        return x