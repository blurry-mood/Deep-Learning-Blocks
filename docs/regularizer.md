# **Anti Correlation**
This regularizer computes the cross-correlation matrix and derives a loss that stifles the model from learning the same thing in differen places at the same layer.

For more insights, check this paper: https://arxiv.org/abs/2103.03230

### **Formula**:
Let z be a matrix of N observations and D variables (i.e each row corresponds to an observation, each column corresponds to a variable) such as the mean and standard deviation of z are 0 and 1 respectively.  
$cross-correlation=\frac{z^Tz}{N}$   
$mm = (cross-correlation - I_D)^2$  
The loss is defined as:  
$loss = sum\_diag(mm) + \lambda *sum\_off\_diag(mm)$

### **How to use**:
1- Import AntiCorrelation module:
> from deepblocks.regularizer import AntiCorrelation

2- Create an instance:
>  loss = AntiCorrelation(p: float = 0.5, lmd: float = 0.05)

- p: Float in [0, 1] denoting the probability of computing the loss with respect to a tensor in the list (check step 3 for more understanding). It's useful since it's not necessary to compute this loss in every forward+backward proagation step, thus it reduces the computational burden.
If p==0, the computed loss value is 0, since there's nothing to compute. 
- lmd: Float mutliplying the off-diagonal sum before adding the diagonal sum: diag_sum + lmd * off_diag_sum

3- Compute the loss:
> _loss = loss(x)

- x (List[Tensor]): The rank of each tensor must be larger than 1, i.e its shape must be [batch, d1, *], albeit it's not necessary for all elements in the list to have the same shape, the list may contain feature maps from a CNN, activations from an MLP extracted from anywhere within the networks.