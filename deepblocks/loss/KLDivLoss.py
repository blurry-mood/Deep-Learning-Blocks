import torch
from torch.nn.functional import log_softmax, logsigmoid, kl_div, softmax, sigmoid
from ._Loss import _Loss

@torch.jit.script
def softmax_kldiv(input:torch.Tensor, y:torch.Tensor, dim:int, reduction:str):
    input = log_softmax(input, dim=dim)
    y = softmax(y, dim=dim)
    loss = kl_div(input, y, reduction=reduction, log_target=False)
    return loss


class KLDivLoss(_Loss):
    """ Initialize a nn.Module that computes the Kullback-Leibler divergence loss.
        It takes two raw tensors, transforms them to probabilities using the softmax activation, then computes the loss.

    Args:
        - dim(Integer, Default=1): The dimension along which Softmax will be computed (so every slice along dim will sum to 1), 
                                     for both the input X and the targets Y.
        - reduction(Str, Default='mean'): Specifies the reduction to apply to the output: 'none' | 'batchmean' | 'sum' | 'mean'. 
                                        'none': no reduction will be applied. 
                                        'batchmean': the sum of the output will be divided by batchsize. 
                                        'sum': the output will be summed. 
                                        'mean': the output will be divided by the number of elements in the output.
    
    Forward args:
        - x(Tensor): Its shape is (N, ...).
        - y(Tensor): The same shape as x, i.e. (N, ...)
    
    """
    
    def __init__(self, dim:int=1, reduction:str='mean'):
        super(KLDivLoss, self).__init__(dim, reduction)

        self.loss = softmax_kldiv

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        return self.loss(x, y, self.dim, self.reduction)
