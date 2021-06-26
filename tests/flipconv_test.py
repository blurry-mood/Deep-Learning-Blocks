''' Test the initialization, forward and backward propagation of the FlipConv2d layer
'''

from deepblocks.CNN.FlipConv2d import FlipConv2d
import torch


def test_init():
    conv = FlipConv2d(h_invariant = False, v_invariant = False,
                        in_channels=4, out_channels=10, kernel_size=5, stride=1, padding=2)
    
    assert conv.kernel.shape == (10, 4, 5, 5)
    assert conv.bias.shape == (10,)
    # Check the number of parameters
    assert sum(p.numel() for p in conv.parameters() if p.requires_grad) == 5**2*4*10 + 10

def test_forward():
    conv = FlipConv2d(h_invariant = False, v_invariant = False, 
                        in_channels=4, out_channels=10, kernel_size=5, stride=1, padding=2)
    img = torch.rand(3, 4, 232, 32)
    y = conv(img)
    assert y.shape == (3, 10, 232, 32)

def test_backward():
    conv = FlipConv2d(h_invariant = False, v_invariant = False,in_channels=4, out_channels=10, kernel_size=3, stride=1, padding=1)
    opt = torch.optim.Adam(conv.parameters(), lr=1e-1)
    img = torch.rand(3, 4, 32, 32)

    for _ in range(1000):
        opt.zero_grad()
        y = conv(img)
        y = y.abs().mean()
        y.backward()
        opt.step()