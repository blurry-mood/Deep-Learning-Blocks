from typing import List

import torch
from torch import nn
from torch.optim import Optimizer

class SAM(Optimizer):
    """ This file contains an implementation of the Sharpness-Aware Minimization optimization algorithm.
    This optimizer is defined with respect to two parameters: 
    - an integer `p` referring to the p-norm used to determine the neighborhood of the model parameters,
    - a float `rho` defining the boundary of the aforementioned neighborhood.

    This optimizer, in contrast with standard ones, relies on two phases to perform a single step:
    1. `first_step`: it computes the position having the worst training loss value in the neighborhood,
    2. `second_step`: it computes the gradient at that calculated position, then takes a step.
    **Note**: Before executing the second step, an additional forward-backward propagation pass must be performed.

    Check the original paper at: https://arxiv.org/abs/2010.01412
    """
    def __init__(self, parameters, base_optimizer:Optimizer, rho:float=0.1, p:int=2, **kwargs):
        """ Initializes the SAM-based optimizer.

        Args:
            - parameters: List of an nn.Module parameters.
            - base_optimizer: The class of the optimizer used to update the weights.
            - rho: Non-negative float defining the boundary of the neighborhood. 
                    Note that using `rho=0` is equivalent to using the bare base_optimizer.
            - p: Positive Integer specifiying the p-norm.
            - **kwargs: additional parameters passed to the base_optimier

        Raises:
            - ValueError: if p is not positive or rho is negative.
        """
        if p<=0:
            raise ValueError(f'The value of p={p} should be a positive integer')
        if rho<0:
            raise ValueError(f'The value of rho={rho} must be a non-negative float')
        q = 1/(1-1/p)
        defaults = dict(rho=rho, p=p, q=q)

        super(SAM, self).__init__(parameters, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups


    @torch.no_grad()
    def first_step(self, zero_grad=True):
        """ Calculate the worst position in the neighborhood.

        Args:
            -zero_grad: Boolean value specifiying whether to zero the gradient after computing the worst position.
                        By default, the gradients are zeroed.
        """
        norm = self._grad_norm()    # This norm is not raised to 1/q

        for group in self.param_groups:
            p, q = group['p'],group['q']
            norm = norm.pow(1/p)
            scale = group['rho']/(norm + 1e-7)

            for param in group['params']:
                if param.grad is None: 
                    continue
                eps = scale * torch.sign(param.grad) * param.grad.abs().pow(q-1)    # Compute where is the worst position direction
                self.state[param]['eps'] = eps  # Save this for the 2nd step
                param.add_(eps)                 # Update position to the worst.

        if zero_grad:
            self.base_optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """ Perform a gradient step

        Args:
            -zero_grad: Boolean value specifiying whether to zero the gradient after computing the worst position.
        """
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None: continue
                param.sub_(self.state[param]['eps'])        # Retrieve old values of parameter

        self.base_optimizer.step()      # Change old weights using SAM loss gradient

        if zero_grad:
            self.base_optimizer.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """ Function that performs both phases of SAM.

        Args:
            - closure: Function with no arguments that takes an additional forward & backward propagation steps.

        Raises:
            - ValueError: if a closure function is not passed.
        """
        if closure is None:
            raise ValueError('You must supply a closure function that calculates the loss of the current batch.')

        closure = torch.enable_grad()(closure)

        self.first_step()
        closure()
        self.second_step()

    @staticmethod
    def closure(model:nn.Module, loss:nn.Module, inputs:List, outputs:List):
        """ A useful function that returns a closure function.

        Args:
            - model: PyTorch module.
            - loss: Loss function taking the predictions of the model and the outputs.
            - inputs: A List of model inputs. The list is unpacked when passed to the model.
            - outputs: A list of ground truth values. The list is unpacked when passed to the loss.

        Returns:
            - closure: A closure function that is passed to SAM.step(closure=closure)
        """
        def _closure():
            preds = model(*inputs)
            _loss = loss(*preds, *outputs)
            _loss.backward()
            return _loss
        return _closure

    def _grad_norm(self):
        q = self.param_groups[0]['q']
        # Compute the q-norm of the model gradient but without raising it to **(1/q)
        return torch.sum(torch.stack(
                                        [param.grad.pow(q).sum() for group in self.param_groups for param in group['params'] if param.grad is not None]
                                     ))