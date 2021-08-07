from torch import nn
from varname import argname2

class _Loss(nn.Module):

    def __init__(self, *args):
        super().__init__()

        names = argname2('*args')
        for k, v in zip(names, args):
            setattr(self, k, v)