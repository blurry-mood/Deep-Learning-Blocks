from typing import Optional
from torch import nn
import torch


class PSigmoid(nn.Module):

    def __init__(self, eps: Optional[int] = None):
        super().__init__()

        self.eps = nn.Parameter(torch.rand(1), requires_grad=True) if eps is None else torch.tensor(eps)

    def forward(self, x):
        return x/(x.abs() + self.eps.abs() + 1e-5)


if __name__ == '__main__':
    with torch.no_grad():
        psig = PSigmoid(1e-7)
        x = torch.randn(2, 3)
        y = psig(x)
        print(x)
        print(y)
