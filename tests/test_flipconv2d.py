import torch
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip
from deepblocks.layer import FlipConv2d

torch.set_printoptions(precision=10)


def test_shapes():
    conv = FlipConv2d(in_channels=3, out_channels=8,
                      kernel_size=3, padding=1, stride=1)
    x = torch.rand(2, 3, 32, 32)
    y = conv(x)
    assert y.shape == (2, 8, 32, 32)

@torch.no_grad()
def test_invariance():
    
    vflip = RandomVerticalFlip(p=1.)
    hflip = RandomHorizontalFlip(p=1.)
    for _ in range(10):
        x = 10*torch.rand(1, 3, 224, 224)
        x = [x, vflip(x), hflip(x), vflip(hflip(x))]

        assert (x[0]==vflip(x[1])).all()
        assert (x[0]==hflip(x[2])).all()
        assert (x[0]==vflip(hflip(x[3]))).all()

        vhconv = FlipConv2d(h_invariant=True, v_invariant=True, in_channels=3,
                            out_channels=1, kernel_size=3, padding=1, stride=1)
        vconv = FlipConv2d(h_invariant=False, v_invariant=True, in_channels=3,
                        out_channels=1, kernel_size=3, padding=1, stride=1)
        hconv = FlipConv2d(h_invariant=True, v_invariant=False, in_channels=3,
                        out_channels=1, kernel_size=3, padding=1, stride=1)

        assert (hconv.kernel == hflip(hconv.kernel)).all()
        assert (vconv.kernel == vflip(vconv.kernel)).all()
        assert (vhconv.kernel == vflip(hflip(vhconv.kernel))).all()

        assert (hconv(x[0]) - hflip(hconv(x[2]))).abs().sum()<1
        assert (vconv(x[0]) - vflip(vconv(x[1]))).abs().sum()<1
        assert (vhconv(x[0]) - vflip(hflip(vhconv(x[-1])))).abs().sum()<1


def test_gradient_flow():
    vflip = RandomVerticalFlip(p=1.)
    conv = FlipConv2d(h_invariant=False, in_channels=1, out_channels=1, kernel_size=5, padding=2, stride=5, bias=True)
    x = .1 + torch.rand(1, 1, 32, 32)
    x = [x, vflip(x)]
    x = torch.cat(x)
    opt = torch.optim.AdamW(conv.parameters(), lr=1e-3)
    _loss = 1e10
    for i in range(1_000):
        opt.zero_grad()
        y = conv(x).abs().sum()
        assert y.item() < _loss + 1
        y.backward()
        opt.step()
        _loss = y.item()