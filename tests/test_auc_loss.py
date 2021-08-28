import torch
from deepblocks.loss import AUCLoss
from torchmetrics import AUROC


def test_shapes():
    auc = AUCLoss(gamma=0.1, reduction='none')
    x = torch.tensor([10, .3, 1, 1]).view(-1, 1)
    y = torch.tensor([1, 1, 0, 0])
    assert auc(x, y).shape == (2, )


def test_consistency():
    x = torch.nn.Parameter(torch.randn(100, 1))
    y = torch.randint(0, 2, size=(100,))
    auc = AUCLoss(gamma=0.1, reduction='mean')
    auroc = AUROC(pos_label=1, compute_on_step=True)
    opt = torch.optim.SGD([x], lr=3)

    _loss, _metric = 1e10, -1e10
    for _ in range(100):
        loss = auc(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        metric = auroc(torch.sigmoid(x), y.unsqueeze(1))

        assert _loss > loss.item()
        assert metric.item() > _metric or _metric >= .99

        _loss = loss.item()
        _metric = metric.item()


def test_skip():
    auc = AUCLoss(gamma=0.1, skip=True, reduction='none')
    x = torch.tensor([10, .3, 1, 1]).view(-1, 1)
    y = torch.tensor([1, 1, 1, 1])
    assert auc(x, y).item() == 1.0
