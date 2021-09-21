from typing import List
import torch
from torch import nn

class AntiCorrelation(nn.Module):
    """ This module stimulates the network to reduce redundancy in its layers.

    It takes a list of tensors (produced by network layers) and returns the loss value: (cross-correlation matrix - Identity)**2
    It uses the parameter lambda to tradeoff between the sum of the diagonal and off-diagonal of the latter matrix.
    (i.e off-diagonal sum multiplied by lambda).

    Note:
        This module is inspired from the Barlow Twin paper: https://arxiv.org/abs/2103.03230

    Args:
        p (float, Optional): Probability of computing the loss with respect to each tensor in the list (to reduce computation).
        lmd (float, Optional): Coefficient mutliplying the off-diagonal sum before adding the diagonal sum: `diag_sum + lmd * off_diag_sum`.
    
    Shape:
        - x (List[torch.Tensor]): The rank of each tensor must be larger than 1,
                               i.e its shape must be [batch, d1, *],
                                otherwise, the last dimension must be unsqueezed before passing it.
                                Also, the first dimension must be equal for every tensor in passed list.

    Example:
        >>> from deepblocks.regularizer import AntiCorrelation
        >>> anti_corr = AntiCorrelation(lmd=1.)
        >>> x1 = torch.rand(100, 24)
        >>> x2 = torch.rand(100, 30, 300)
        >>> loss = anti_corr([x1, x2])
    """

    def __init__(self, p: float = 0.5, lmd: float = 0.05):
        super().__init__()

        self.p = p
        self.lmd = lmd
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x: List[torch.Tensor]):
        """ """
        xx = []
        for t in x:
            if torch.rand(1) < self.p:
                # Random variable at the last index
                xx.append(t.flatten(1))

        if xx == []:
            return torch.tensor(0, device=self.dummy_param.device)

        x = torch.cat(xx, dim=-1)  # N x D

        # Normalize
        z = (x - x.mean(0)) / x.std(0)

        N, D = z.shape
        mm = torch.mm(z.T, z) / N  # D x D; D: number of RVs.

        # Diagonal entries must be equal to 1; since normalized
        mm = (mm - torch.eye(D, device=z.device, requires_grad=False)).pow(2)

        # Extract diagonal & off-diagonal
        diag = torch.diag(torch.diag(mm))
        off_diag = mm - diag

        return diag.sum() + off_diag.sum() * self.lmd