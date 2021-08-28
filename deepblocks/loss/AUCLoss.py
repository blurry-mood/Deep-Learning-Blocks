import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy

from ._Loss import _Loss


@torch.jit.script
def _binary_loss(x: torch.Tensor, y: torch.Tensor, gamma: float, p: float, skip: bool):
    x = torch.sigmoid(x)
    y = y.unsqueeze(1)     # make it broadcastable

    # compute masks
    pos, neg = [y == i for i in [1, 0]]
    # compute masked values
    pos, neg = [torch.masked_select(x, _x) for _x in [pos, neg]]
    if pos.shape == (0,) or neg.shape == (0,):
        return torch.tensor(1., dtype=torch.float32, requires_grad=True, device=y.device) if skip else binary_cross_entropy(x, y.float(), reduction='none')

    # compute difference
    diff = pos.unsqueeze(0) - neg.unsqueeze(1) - gamma
    # compute mask for diff<gamma
    mask = diff < 0
    diff = torch.masked_select(diff, mask)
    if diff.shape == (0, ):
        return torch.tensor(1., dtype=torch.float32, requires_grad=True, device=y.device) if skip else binary_cross_entropy(x, y.float(), reduction='none')
    # compute loss
    loss = (-diff).pow(p)
    return loss


class AUCLoss(_Loss):
    """ This loss is an approximation to WilcoxonMann-Whitney (WMW) statistic. 
        It's used to directly maximize the Area Under the Curve (AUC) metric. 
        Note that each batch must have both positive and negative samples and 
        at least a pair of positive & negative samples must satisfy the constraint (pos_i - neg_j < gamma) 
        to compute this loss, otherwise, the weights aren't updated if `skip==True` else the BCELoss is computed for that batch.
        Only the binary classification case is supported by this implementation.

        For more details check: https://www.aaai.org/Papers/ICML/2003/ICML03-110.pdf

    Args:
        - gamma (Float, Default=0.3): denotes the tolerance in the difference between positive and negative samples, 
                                    if not exceeded that pair doesn't contribute to the loss. 
                                    The larger the `gamma`, the more the model enlarges the gap between positive and negative predictions
        - p (Float, Default=2.0): denotes the exponent used for the aforementioned difference.
        - skip (Boolean, Default=False): specifies whether to skip, if true, the current weight update step or to use the BCELoss function instead if false, 
                                     when the current batch doesn't have both positive and negative samples or none of the pairs satisfy (pos_i - neg_j < gamma).
        - reduction(String, Default='mean'): specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 
                                    'none': no reduction will be applied, 
                                    'mean': the sum of the output will be divided by the number of elements in the output, 
                                    'sum': the output will be summed.

    Forward Args:
        - x (Tensor): It is a (N, 1) tensor with predictions before transforming them to probabilities.
        - y (Tensor): It is a (N, ) tensor with ground truth labels, meaning zeros and ones.

    """

    def __init__(self, gamma: float = 0.3, p: float = 2, skip: bool = False, reduction: str = 'mean'):
        super(AUCLoss, self,).__init__(gamma, p, skip, reduction)
        if not (0 < gamma <= 1):
            raise ValueError(
                f'The value of gamma={gamma} must be in (0, 1]')
        if p <= 1:
            raise ValueError(
                f'The value of p={p} must larger than 1')

        self.loss = _binary_loss

    def forward(self, x: Tensor, y: Tensor):
        loss = self.loss(x, y, self.gamma, self.p, self.skip)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss