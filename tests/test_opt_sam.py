from torch import nn
from copy import deepcopy
from deepblocks.optim import SAM

import torch


class _SAM(torch.optim.Optimizer):
    """ From: https://github.com/davda54/sam """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(_SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2)
                       if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # the closure should do a full forward-backward pass
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0)
                         * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                        ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def test_loss_decrease():
    model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 5))
    sam = SAM(model.parameters(), torch.optim.SGD, rho=0.05, p=2, lr=1e-2)
    x = torch.randn(10, 10)
    y = 10*torch.randn(10, 5)

    _loss = 1e10
    for _ in range(10):
        sam.zero_grad()
        y_hat = model(x)
        loss = (y-y_hat).pow(2).mean()

        # assert & update best loss
        assert loss.item() < _loss
        _loss = loss.item()

        loss.backward()
        sam.first_step()
        y_hat = model(x)
        (y-y_hat).pow(2).mean().backward()
        sam.second_step()

def test_against_other_implementation():
    model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 5))
    _model = deepcopy(model)
    
    sam = SAM(model.parameters(), torch.optim.SGD, rho=0.05, p=2, lr=1e-2)
    _sam = _SAM(_model.parameters(), torch.optim.SGD, rho=0.05, lr=1e-2)
    
    x = torch.randn(10, 10)
    y = 10*torch.randn(10, 5)


    for i in range(10):
        # deepblocks implementation
        sam.zero_grad()
        loss = (y-model(x)).pow(2).mean()
        loss.backward()
        sam.first_step()
        (y-model(x)).pow(2).mean().backward()
        sam.second_step()

        # other implementation
        _loss = (y-_model(x)).pow(2).mean()
        _loss.backward()
        _sam.first_step(True)
        (y-_model(x)).pow(2).mean().backward()
        _sam.second_step(True)

        # assertion
        for param, _param in zip(model.parameters(), _model.parameters()):
            assert loss - _loss<1e-5, i
            assert (param.data - _param.data).abs().sum()<1e-5, i