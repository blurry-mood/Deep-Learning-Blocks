# Focal Loss
This loss is derived from the cross entropy loss with the aim of reducing the dominance of easy examples (proba > 0.5) on the loss value, thus highlighting the loss of hard examples (proba < 0.5).

This loss is defined with respect to a float hyperparammeter `gamma` that is non-negative. It controls how 
the modulating factor attenuate the loss of easy examples, larger values of `gamma` decrease their loss value.

Moreover, another parameter (tensor) `alpha` could be used to further balance the loss of each sample when computing the loss.

For more details check: https://arxiv.org/abs/1708.02002

### **Formula**:
For binary classification, the loss is defined as:   
$loss(x_t,y_t) = - \alpha_t  (y_t.  log(\sigma(x_t)).(1-\sigma(x_t))^\gamma + (1-y_t).log(1-\sigma (x_t)).\sigma(x_t)^
\gamma)$

For multi-class classification,  the loss is defined as:
$loss(x_t,y_t) = - \alpha_t . log(softmax(x_t)_{y_t}) (1-softmax(x_t)_{y_t})^\gamma$
  


### **How to use**:
1- Import the focal loss:
> from deepblocks.loss import FocalLoss

2- Create an instance:
>  loss = FocalLoss(cls: bool = False, gamma: float = 2., reduction: str = 'mean')
* `cls` (Boolean, Default=False): denotes whether the loss is used in the context of binary classification/multi-labels, or in the context of classification.
* `gamma` (Float, Default=2): the factor attenuating the magnitude of easy examples. It should be non-negative.
* `reduction`(String, Default='mean'): specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 
                            'none': no reduction will be applied, 
                            'mean': the sum of the output will be divided by the number of elements in the output, 
                            'sum': the output will be summed.

3- Compute the loss: 
> loss(x, y, alpha=None)
* `x` (Tensor): If `cls` is set to false, x is a (N, *) tensor with predictions before transforming them to probabilities. If `cls` is set to true, x is a (N, C, d1, *) tensor, it's softmaxed along the dimension 1 (dimension start from 0).
* `y` (Tensor): If `cls` is set to false, y is a (N, *) float tensor with ground truth labels. If `cls` is set to true, x is a (N, d1, *) long tensor of ground truth labels.
* `alpha` (Tensor, Optional): A float tensor with the same shape as y, or broadcastable.