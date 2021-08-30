import torch
from deepblocks.loss import AUCLoss
from torchmetrics import AUROC


def test_shapes():
    auc = AUCLoss(gamma=0.01, reduction='none')
    x = torch.tensor([10, .3, 1, 1]).view(-1, 1)
    y = torch.tensor([1, 1, 0, 0]).view(-1, 1)
    assert auc(x, y).shape == (10, )


def test_consistency():
    # decrease in loss <=> increase in auroc
    x = torch.nn.Parameter(torch.randn(100, 1))
    y = torch.randint(0, 2, size=(100,)).view(-1, 1)
    auc = AUCLoss(gamma=0.1, reduction='mean')
    auroc = AUROC(pos_label=1, compute_on_step=True)
    opt = torch.optim.SGD([x], lr=9e-1)

    _loss = auc(x, y).item()
    _metric = auroc(torch.sigmoid(x), y).item()

    print('initial loss value:',_loss, ',  initial auc value:', _metric)

    for _ in range(100):
        loss = auc(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        metric = auroc(torch.sigmoid(x), y)

        assert _loss >= loss.item() and metric.item() >= _metric

        _loss = loss.item()
        _metric = metric.item()
    print('final loss value:',_loss, ',  final auc value:', _metric)


def test_homogenous_batch():
    # if all samples belong the same class 
    auc = AUCLoss(gamma=0.01, reduction='none')
    x = torch.tensor([10, .3, 1, 1]).view(-1, 1)
    y = torch.tensor([1, 1, 1, 1]).view(-1, 1)
    assert auc(x, y).item() == 0.0
