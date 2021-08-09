from deepblocks.ssl import dino

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip


class Module(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(4**2, 32)
        )

    def forward(self, x):
        return self.model(x)


def test_from_bugs():
    model = Module()
    dataloader = DataLoader([torch.rand(3, 32, 32)
                             for _ in range(100)], batch_size=16, num_workers=4)
    pretrained = dino(model, output_dim=32,
                dataloader=dataloader, 
                global_crops_scale=(.1, .9), local_crops_scale=(.1, .9), local_crops_number=9,
                opt='Adam', opt_kwargs={'lr': 1e-3},
                max_epochs=3, gpus=0, )

    assert isinstance(pretrained, Module)

if __name__=='__main__':
    test_from_bugs()