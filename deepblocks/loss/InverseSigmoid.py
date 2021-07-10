''' Implementation of loss function that's similar to the inverse of the sigmoid function.
    It could be used instead of CrossEntropy loss function.
'''

from typing import Optional
from numpy import log
import torch
from torch import nn
from torch.nn.functional import nll_loss, softmax

@torch.jit.script
def invsig_loss(y:torch.Tensor, inds:torch.Tensor, alpha:float, beta:float, gamma:float, reduction:str):
    y = softmax(y, dim=1)
    y = - nll_loss(y, inds, reduction='none')
    y = torch.log(1 / (alpha * y + beta) - 1) + gamma
    if reduction == 'sum':
        return y.sum()
    elif reduction == 'mean':
        return y.mean()
    return y

class InverseSigmoid(nn.Module):
    
    def __init__(self, alpha: float = .9, beta: float = .043, reduction:Optional[str]='mean'):
        ''' Initialisation of the loss function.
            The formula is :
                loss = torch.log(1 / (alpha * y + beta) - 1) + gamma
            where: gamma is a float that makes sure the loss is always positive, with loss(1)==0.
        Args: 
            - alpha, beta (floats): Two positive floats used to control: gradient step near 0 and 1.
                                    They need to be set carefully; always make sure that: 'alpha + beta < 1.'
            - reduction (string, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 
                                            'none': no reduction will be applied, 
                                            'mean': the mean of the output is taken, 
                                            'sum': the output will be summed.
        '''
        super().__init__()

        # assert alpha>0 and beta>0 and alpha+beta<1, f'Make sure: {alpha=}>0, {beta=}>0, and {alpha+beta=}<1 '

        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        # Assert loss(1)==0
        self.gamma = -log(1/(alpha + beta)-1)

    def forward(self, y:torch.Tensor, inds:torch.Tensor):
        ''' Compute the inverse sigmoid loss function
        
        Args:
            - y: A Float tensor with a shape (batch, C, d1, d2, ..., dK), C: number of classes.
            - inds: A Long tensor of indices, with a shape (batch, d1, d2, ..., dK), 
                    where each entry is non-negative and smaller than C.
        
        Returns:
            - loss: depending on the reduction, the resulting tensor could be a scalar tensor, 
                    or a tensor with the same shape as 'inds' if 'reduction' is 'none'.
        '''
        return invsig_loss(y, inds, self.alpha, self.beta, self.gamma, self.reduction)