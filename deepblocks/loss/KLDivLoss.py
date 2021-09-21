import torch
from torch.nn.functional import log_softmax, kl_div, softmax
from ._Loss import _Loss

@torch.jit.script
def _softmax_kldiv(input:torch.Tensor, y:torch.Tensor, dim:int, reduction:str):
    input = log_softmax(input, dim=dim)
    y = softmax(y, dim=dim)
    loss = kl_div(input, y, reduction=reduction, log_target=False)
    return loss


class KLDivLoss(_Loss):
    """ Initialize a nn.Module that computes the Kullback-Leibler divergence loss.

    It takes two logits tensors, transforms them to probabilities using the softmax activation, then computes the loss.
    
    Note:
        This implementation supports Softmax as a mean to transform logits to probabilities.

    Args:
        dim (int, Optional): The dimension along which Softmax will be computed (so every slice along `dim` will sum to 1), 
                              for both `x` and `y`. Default is 1.
        reduction (str, Optional): Specifies the reduction to apply to the output: 'none' | 'batchmean' | 'sum' | 'mean'. 
                                        'none': no reduction will be applied. 
                                        'batchmean': the sum of the output will be divided by batchsize. 
                                        'sum': the output will be summed. 
                                        'mean': the output will be divided by the number of elements in the output.
                                        Default is 'batchmean`.
    
    Shape:
        - x (torch.Tensor): Its shape is (N, ...). It's the tensor of logits that is desired to match.
        - y (torch.Tensor): The same shape as x, i.e. (N, ...). It's the tensor of logits that should match the distribution of `x`.
    
    Example:
        >>> from deepblocks.loss import KLDivLoss
        >>> kldiv = KLDivLoss()
        >>> x = torch.randn(10, 20, 33)
        >>> y = torch.rand(10, 20, 33)
        >>> loss = kldiv(x, y)  # loss.shape = ()
    
    """
    
    def __init__(self, dim:int=1, reduction:str='batchmean'):
        super(KLDivLoss, self).__init__(dim, reduction)

        self.loss = _softmax_kldiv

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        """ """
        return self.loss(x, y, self.dim, self.reduction)
