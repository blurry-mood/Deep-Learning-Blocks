from deepblocks.loss import FocalLoss
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

def test_shapes():
    # Testing loss in ctx of multi-classification & multi-labels, with different reductions
    x = torch.rand(3, 5, 20, 30, 3)
    y_lab = (torch.rand(3, 5, 20, 30, 3) > .5).int()
    y_cls = torch.randint(low=0, high=5, size=(3, 20, 30, 3))
    alpha_lab = torch.rand(3, 1, 1, 1, 1)
    alpha_cls = torch.rand(3, 1, 1, 1)
    # Test
    for cls, y, alpha in zip([True, False], [y_cls, y_lab], [alpha_cls, alpha_lab]):
        for red in ['none', 'mean', 'sum']:
            loss = FocalLoss(cls=cls, reduction=red)
            _y = loss(x, y)
            _ya = loss(x, y, alpha)
            if red == 'none':
                assert _y.shape == y.shape
                assert _ya.shape == y.shape
            else:
                assert len(_y.shape) == 0
                assert len(_ya.shape) == 0


def test_consistency_gamma_zero():
    # Testing loss range (x=100, labels=1, cls=0) in ctx of multi-classification & multi-labels, with different reductions
    ones = torch.ones(3, 5, 20, 30, 3)
    y_lab = torch.ones(3, 5, 20, 30, 3)
    y_cls = torch.zeros(3, 20, 30, 3).long()
    # Test
    for func, cls, y in zip([cross_entropy, binary_cross_entropy_with_logits], [True, False], [y_cls, y_lab]):
        # Test different values for x
        for _ in range(200):
            x = torch.randint(low=-100_000_000, high=100_000_000, size=(1,)) * ones 
            loss = FocalLoss(cls=cls, gamma=0)
            _y = loss(x, y)
            assert abs(_y.item() - func(x, y).item())<=1e-4

def test_evolution():
    # Test gradient propagation on a simple example for different values of gamma
    y_cls = torch.tensor([0, 2, 1])
    y_lab = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    for cls, y in zip([True, False], [y_cls, y_lab]):
        x = torch.nn.Parameter(torch.tensor([[-10, 10., 50], [-1, 20, -50], [10, -4, -4]]), requires_grad=True)
        for gamma in range(10):
            loss = FocalLoss(cls=cls, gamma=gamma, reduction='mean')
            opt = torch.optim.AdamW([{'params':x}], lr=1e-1)
            _loss= 1e10
            for _ in range(10):
                opt.zero_grad()
                _tmp = loss(x, y)
                assert 0 < _tmp.item() <= _loss
                _tmp.backward()
                opt.step()
                _loss = _tmp.item()
            assert (x[[0, 1, 2], [0, 2, 1]]> torch.tensor([-10, -50, -4])).all()