""" This is an implementation of two loss functions designed to directly optimize the AUROC metric.

Classes:
    - AUCLoss
    - AUCMarginLoss
"""

import torch
from torch import Tensor

from torch import nn

from ._Loss import _Loss

def _sigmoid(pos: torch.Tensor, neg: torch.Tensor, gamma:float, p:float):
    diff = neg - pos
    return torch.sigmoid(diff)

def _tanh(pos: torch.Tensor, neg: torch.Tensor, gamma:float, p:float):
    diff = neg - pos
    return (torch.tanh(diff)+1)/2

def _power_p(pos: torch.Tensor, neg: torch.Tensor, gamma:float, p:float):
    diff = neg + gamma - pos
    return torch.relu(diff).pow(p)


_FUNCTIONS = {'sigmoid':_sigmoid, 'tanh':_tanh, 'power_p': _power_p}


class AUCLoss(_Loss):
    """ This pairwise loss leverages an approximation to WilcoxonMann-Whitney (WMW) statistic to directly maximize the Area Under the Curve (AUC) metric.

    Note: 
        For more details check: https://www.aaai.org/Papers/ICML/2003/ICML03-110.pdf

    Note: 
        Only the binary classification case is supported by this implementation.

    Args:
        loss (str, Optional): Funtion that computest the distance between a positive-negative pair of samples. 
                    The available options are `sigmoid`, `tanh` & `power_p`. `power_p(negative, positive, gamma) = (negative + gamma - positive)^p.
                    If another string or nothing is passed `power_p`is used by default.
        gamma (float, Optional): Specifies the desired margin between the maximum probability of a negative sample and the minimum probability of a positive sample. 
                        The larger the `gamma`, the more the model enlarges the gap between positive and negative predictions.
                        It should non-negative and less than or equals to 1. This argument is dropped unless `loss` is `power_p`.
                        Default is 0.3.
        p (float, Optional): Specifies the exponent used in `power_p`. It should be larger than 1. This argument  is dropped unless `loss` is `power_p`.
                        Default is 2.
        reduce (str, Optional): Specifies the reduction to apply to the output: 'mean' | 'sum'.  
                    'mean': the sum of the output will be divided by the number of elements in the output, 
                    'sum': the output will be summed.  
                    Default is 'mean'.

    Shape:
        - x (torch.Tensor): It is a (N, *dims) tensor of logits.
        - y (torch.Tensor): It is a (N, *dims) tensor with ground truth labels; meaning zeros and ones.

    Example:
        >>> from deepblocks.loss import AUCLoss
        >>> aucloss = AUCLoss(gamma=0.9, p=1.1, reduce='sum' )
        >>> logits = torch.rand(300, 10, 20)
        >>> labels = torch.randint(low=0, high=2, size=(300, 10, 20))
        >>> loss = aucloss(logits, labels)

    """

    def __init__(self, loss:str='power_p', gamma: float = .3, p: float = 2, reduce:str='mean'):
        super(AUCLoss, self).__init__(gamma, p, reduce)

        if gamma < 0 or gamma>1 :
            raise ValueError(
                f'The value of gamma={gamma} must non-negative and smaller than 1')
        if p <= 1:
            raise ValueError(
                f'The value of p={p} must larger than 1')

        self.func = _FUNCTIONS.get(loss, _power_p)

    def forward(self, x: Tensor, y: Tensor):
        """ """
        # x.shape = y.shape = (batch, *dims)

        x = torch.sigmoid(x)

        x, y = x.flatten(1), y.flatten(1)   # 1st dim: batch

        loss = 0
        for i in range(x.size(1)):
            xx, yy = x[:,i], y[:,i]
            pos = torch.masked_select(xx, yy==1).unsqueeze(0)
            neg = torch.masked_select(xx, yy==0).unsqueeze(1)

            if (pos.nelement() == 0 or neg.nelement() == 0) and loss==0:
                loss += torch.tensor(0., requires_grad=True)
                continue
            
            diff = self.func(pos, neg, self.gamma, self.p )
            loss += diff.mean() if self.reduce=='mean' else diff.sum()

        return loss


class AUCMarginLoss(_Loss):
    """ This is an implementation of the AUC margin loss published in this paper https://arxiv.org/abs/2012.03173

    Note: 
        Only the binary classification case is supported by this implementation.

    Args:
        p (float, Optional): The probability of positive class: P(class=1).   
                    Default is 0.5.
        m (float, Optional): Specifies the desired margin between the probabilities of positive samples and negative samples. 
                        It should non-negative and less than or equals to 1.   
                        Default is 1.
        reduce (str, Optional): Specifies the reduction to apply to the output: 'mean' | 'sum' | anything else.  
                    'mean': the sum of the output will be divided by the number of elements in the output, 
                    'sum': the output will be summed,
                    otherwise, the output will have the same shape as x.   
                    Default is 'mean'.

    Shape:
        - x (torch.Tensor): It is a (N, *dims) tensor of logits.
        - y (torch.Tensor): It is a (N, *dims) tensor with ground truth labels; meaning zeros and ones.

    Example:
        >>> from deepblocks.loss import AUCMarginLoss
        >>> aucmarginloss = AUCMarginLoss(p=0.9, m=0.8)
        >>> logits = torch.rand(300, 10, 20)
        >>> labels = torch.randint(low=0, high=2, size=(300, 10, 20))
        >>> loss = aucmarginloss(logits, labels)

    """
    def __init__(self, p:float=0.5, m:float=1., reduce:str='mean'):
        super(AUCMarginLoss, self).__init__(p, m, reduce)
        if not(0<=p<=1):
            raise ValueError(f'p={p} is probability value, it should belong to [0, 1]')
        if not(0<=m<=1):
            raise ValueError(f'm={m} should belong to [0, 1]')
        
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x, y):
        """ """
        positive = y==1
        negative = y==0

        x = torch.sigmoid(x)

        # alpha must always be nonnegative
        p, a, b, m = self.p, self.a, self.b, self.m
        alpha = torch.relu(1 + b - a)

        loss = (1-p)*(x-a).pow(2)*positive
        loss += p*(x-b).pow(2)*negative
        loss -= p*(1-p)*alpha.pow(2)
        loss += 2*alpha*(p*(1-p)*m + p*x*negative - (1-p)*x*positive)

        if self.reduce=='mean':
            loss = loss.mean()
        elif self.reduce=='sum':
            loss = loss.sum()
        
        return loss

if __name__ == '__main__':
    auc = AUCMarginLoss() # AUCLoss()
    x = torch.rand(10, 1)
    y = torch.randint(0, 1, size=(10, 1))
    auc(x, y)