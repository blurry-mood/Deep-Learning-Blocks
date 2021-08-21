import torch
from deepblocks.loss import KLDivLoss


def test_shape():
    x, y = torch.rand(3, 10, 20), torch.rand(3, 10, 20)
    for red, shape in zip(['none', 'mean', 'sum', 'batchmean'], [(3, 10, 20), (), (), ()]):
        kl = KLDivLoss(reduction=red)
        assert kl(x, y).shape == shape
