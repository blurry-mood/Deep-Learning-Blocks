# **Sharpness-Aware Minimization (SAM)**
This optimization procedures seeks parameters that lie in neighborhoods having uniformly low loss. It's performed in a two-steps algorithm.

For more insights, check this paper: https://arxiv.org/abs/2010.01412

### **Algorithm**:
Check **Algorithm 1** in the original paper.

### **How to use**:
1. Import SAM module:
> from deepblocks.optim import SAM

2. Create an instance:
>  opt = SAM(parameters, base_optimizer:Optimizer, rho:float=0.1, p:int=2, **kwargs)

- `parameters`: iterable of parameters to optimize or dicts defining parameter groups.
- `base_optimizer`: PyTorch optimizer class.
- `rho`: positive float defining the boundary of the neighborhood.
- `p`: positive integer used to define the p-norm.
- `**kwargs`: sequence of keyword arguments passed to the `base_optimizer`. For e.g: *lr*, *weight_decay*, ...

3. Perform first step:  
Computes the value of `eps_hat`.
> opt.first_step(x, zero_grad=True)

4. Perform second step:  
Updates the model parameters.
> opt.second_step(x, zero_grad=False)

### **Example**:
Inside the training loop:
> x, y = batch  
> y_hat = model(x)  
> _loss = loss(y_hat, y)  
> **opt.zero_grad()**  
> _loss.backward()  
> **opt.first_step()**  
> y_hat = model(x)  
> _loss = loss(y_hat, y)  
> _loss.backward()  
> **opt.second_step()**
