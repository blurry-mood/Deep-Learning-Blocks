import torch
from torch import Tensor

from ._Loss import _Loss


# @torch.jit.script
def _binary_loss(x: torch.Tensor, y: torch.Tensor, gamma: float, p: float):
    # x.shape = y.shape = (batch, *dims)

    x = torch.sigmoid(x)

    x, y = x.flatten(1), y.flatten(1)   # 1st dim: batch

    loss = 0
    for i in range(x.size(1)):
        xx, yy = x[:,i], y[:,i]
        pos = torch.masked_select(xx, yy==1)
        neg = torch.masked_select(xx, yy==0)

        if pos.nelement() == 0 or neg.nelement() == 0:
            loss += torch.tensor(0., requires_grad=True)
            continue
        diff = neg.unsqueeze(1) + gamma - pos.unsqueeze(0)
        loss += torch.relu(diff).pow(p).sum()

    return loss


class AUCLoss(_Loss):
    """ This loss leverages an approximation to WilcoxonMann-Whitney (WMW) statistic to directly maximize the Area Under the Curve (AUC) metric.

        Please note that only the binary classification case is supported by this implementation.

        For more details check: https://www.aaai.org/Papers/ICML/2003/ICML03-110.pdf

    Args:
        - gamma (Float, Default=1.): denotes the tolerance in the difference between positive and negative samples predictions, 
                                    if not exceeded that pair doesn't contribute to the loss. 
                                    The larger the `gamma`, the more the model enlarges the gap between positive and negative predictions
        - p (Float, Default=2.0): denotes the exponent used for the aforementioned difference.

    Forward Args:
        - x (Tensor): It is a (N, *dims) tensor with predictions before transforming them to probabilities.
        - y (Tensor): It is a (N, *dims) tensor with ground truth labels; meaning zeros and ones.

    """

    def __init__(self, gamma: float = .3, p: float = 2):
        super(AUCLoss, self,).__init__(gamma, p)

        if gamma < 0:
            raise ValueError(
                f'The value of gamma={gamma} must non-negative')
        if p <= 1:
            raise ValueError(
                f'The value of p={p} must larger than 1')

        self.loss = _binary_loss

    def forward(self, x: Tensor, y: Tensor):
        loss = self.loss(x, y, self.gamma, self.p)
        return loss
