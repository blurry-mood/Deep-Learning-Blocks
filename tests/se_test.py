''' Test the initialization, forward and backward propagation of the Squeeze-Excitation block
'''

from deepblocks.CNN.SE import SE
import torch


def test_init():
    in_channels, ratio = torch.randint(low=1, high=100, size=(2,)).tolist()
    se = SE(in_channels, ratio)

    # Check the number of parameters
    tmp = in_channels//ratio
    assert sum(p.numel() for p in se.parameters(
    ) if p.requires_grad) == (in_channels+1) * tmp + (tmp+1) * in_channels


def test_forward():
    in_channels, ratio = torch.randint(low=1, high=100, size=(2,)).tolist()
    ratio = ratio if ratio<=in_channels else in_channels
    se = SE(in_channels, ratio)

    img = torch.rand(3, in_channels, 32, 32)
    y = se(img)
    assert y.shape == (3, in_channels, 32, 32)


def test_backward():
    in_channels, ratio = torch.randint(low=1, high=100, size=(2,)).tolist()
    ratio = ratio if ratio<=in_channels else in_channels
    se = SE(in_channels, ratio)
    opt = torch.optim.Adam(se.parameters(), lr=1e-1)
    img = torch.rand(3, in_channels, 32, 32)

    for _ in range(10):
        opt.zero_grad()
        y = se(img)
        y = y.abs().mean()
        y.backward()
        opt.step()
