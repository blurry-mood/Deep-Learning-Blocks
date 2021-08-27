import torch
from torch import Tensor

from ._Loss import _Loss


@torch.jit.script
def _binary_loss(x: torch.Tensor, y: torch.Tensor, gamma: float, p: float):
    x = torch.sigmoid(x)
    y = y.unsqueeze(1)     # make it broadcastable

    # compute masks
    pos, neg = [y == i for i in [1, 0]]
    # compute masked values
    pos, neg = [torch.masked_select(x, _x) for _x in [pos, neg]]
    # compute difference
    diff = pos.unsqueeze(0) - neg.unsqueeze(1) - gamma
    # compute mask for diff<gamma
    mask = diff < 0
    diff = torch.masked_select(diff, mask)
    # compute loss
    loss = (-diff).pow(p)
    return loss


class AUCLoss(_Loss):
    """ This loss is an approximation to WilcoxonMann-Whitney (WMW) statistic. 
        It's used to directly maximize the Area Under the Curve (AUC) metric.
        Only the binary classification case is supported by this implementation.
        
        For more details check: https://www.aaai.org/Papers/ICML/2003/ICML03-110.pdf

    Args:
        - gamma (Float, Default=0.3): denotes the tolerance in the difference between positive and negative samples, 
                                    if not exceeded that pair doesn't contribute to the loss.
        - p (Float, Default=2.0): denotes the exponent used for the aforementioned difference.
        - reduction(String, Default='mean'): specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 
                                    'none': no reduction will be applied, 
                                    'mean': the sum of the output will be divided by the number of elements in the output, 
                                    'sum': the output will be summed.

    Forward Args:
        - x (Tensor): It is a (N, 1) tensor with predictions before transforming them to probabilities.
        - y (Tensor): It is a (N, ) tensor with ground truth labels, meaning zeros and ones.

    """

    def __init__(self, gamma: float = 0.3, p: float = 2, reduction: str = 'mean'):
        super(AUCLoss, self,).__init__(gamma, p, reduction)
        if not (0 < gamma <= 1):
            raise ValueError(
                f'The value of gamma={gamma} must be in (0, 1]')
        if p <= 1:
            raise ValueError(
                f'The value of p={p} must larger than 1')

        self.loss = _binary_loss

    def forward(self, x: Tensor, y: Tensor):
        loss = self.loss(x, y, self.gamma, self.p)
        if self.reduction  == 'mean':
            return loss.mean()
        elif self.reduction =='sum':
            return loss.sum()
        return loss
