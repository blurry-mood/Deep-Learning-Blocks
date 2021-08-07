from typing import Optional

from torch import Tensor
import torch
from ._Loss import _Loss
from torch.nn.functional import softmax, nll_loss, logsigmoid, log_softmax


@torch.jit.script
def cls_loss(x: Tensor, y: Tensor, gamma: float):
    x = log_softmax(x, dim=1)                   # Compute probabilities
    x = -nll_loss(x, y, reduction='none')    # Mask negative classes
    xx = 1 - x.exp()                              # Compute modulating factor
    l = -xx.pow(gamma) * x           # Compute loss
    return l


def label_loss(x: Tensor, y: Tensor, gamma: float):
    s_x = torch.sigmoid(x)    # Compute probabilities
    s_xx = 1 - s_x                # Compute modulating factor
    l_xx = logsigmoid(1-x)
    l_x = logsigmoid(x)
    yy = 1-y                # Compute 'mask'
    l = yy*s_x.pow(gamma)*l_xx + y*s_xx.pow(gamma)*l_x
    return -l


class FocalLoss(_Loss):
    """ This loss is derived from the cross entropy loss with the aim of reducing the dominance of easy examples (proba > 0.5)
        on the loss value, thus highlighting the loss of hard examples (proba < 0.5).

        This loss is defined with respect to a float hyperparammeter `gamma` that is non-negative. It controls how 
        the modulating factor attenuate the loss of easy examples, larger values of `gamma` decrease their loss value.

        Moreover, another parameter (tensor) `alpha` could be used to further balance the loss of each sample when computing the loss.

        For more details check: https://arxiv.org/abs/1708.02002

    Args:
        - cls (Boolean, Default=False): denotes whether the loss is used in the context of binary classification/multi-labels, 
                                        or in the context of classification.
        - gamma (Float, Default=2): the factor attenuating the magnitude of easy examples. It should be non-negative.
        - reduction(String, Default='mean'): specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 
                                    'none': no reduction will be applied, 
                                    'mean': the sum of the output will be divided by the number of elements in the output, 
                                    'sum': the output will be summed.

    Forward Args:
        - x (Tensor): If `cls` is set to false, x is a (N, *) tensor with predictions before transforming them to probabilities. If `cls` is set to true, x is a (N, C, d1, ...) tensor, it's softmaxed along the dimension 1 (dimension start from 0).
        - y (Tensor): If `cls` is set to false, y is a (N, *) float tensor with ground truth labels. If `cls` is set to true, x is a (N, d1, ...) long tensor of ground truth labels.
        - alpha (Tensor, Optional): A float tensor with the same shape as y, or broadcastable.

    """

    def __init__(self, cls: bool = False, gamma: float = 2., reduction: str = 'mean'):
        super(FocalLoss, self,).__init__(gamma, reduction)
        if gamma < 0:
            raise ValueError(
                f'The value of gamma={gamma} must always be non-negative.')

        # choose the loss according to the 'cls' argument
        self.loss = cls_loss if cls else label_loss

    def forward(self, x: Tensor, y: Tensor, alpha: Optional[Tensor] = None):
        l = self.loss(x, y, self.gamma)             # Compute the loss

        if alpha is not None:
            l = alpha * l                           # Balance loss

        # Reduce the loss
        if self.reduction == 'mean':
            l = l.mean()
        elif self.reduction == 'sum':
            l = l.sum()

        return l
