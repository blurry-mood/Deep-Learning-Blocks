from deepblocks.regularizer import AntiCorrelation

import torch
from torch import nn

torch.set_printoptions(precision=10)


def test_from_bug():
    anti = AntiCorrelation(p=0., lmd=1.)
    x = torch.rand(10, 3)
    y = anti([x])

    assert len(y.shape) == 0


def test_break_symmetry():
    param = nn.Parameter(torch.ones(3, 4))
    opt = torch.optim.Adam([{'params': param}], lr=1e-1)
    x = torch.rand(10, 3)
    for _ in range(100):
        y = x@param
        y = y.mean()
        opt.zero_grad()
        y.backward()
        opt.step()
        assert (param[1:]-param[:-1]).abs().sum() < 1e-1

    anti = AntiCorrelation(p=1., lmd=.01)
    param = nn.Parameter(torch.ones(3, 4))
    opt = torch.optim.Adam([{'params': param}], lr=1e-1)
    for _ in range(100):
        y = x @ param
        loss = anti([y])
        y = y.mean() + loss
        opt.zero_grad()
        y.backward()
        opt.step()
    assert (param[1:]-param[:-1]).abs().mean() > 1e-1
