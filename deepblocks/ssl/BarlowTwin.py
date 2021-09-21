from typing import Dict, List, Union

import torch
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, LightningDataModule


_opts = {'Adam': Adam, 'AdamW': AdamW, 'SGD': SGD,
         'RMSprop': RMSprop, 'Adagrad': Adagrad}


@torch.jit.script
def _loss(z1: torch.Tensor, z2: torch.Tensor, lmd: float):
    # z1 &  z2 must be two matrices: 2D tensor
    N, D = z1.shape
    mm = torch.mm(z1.T, z2) / N  # DxD

    mm = (mm - torch.eye(D, device=z1.device)).pow(2)

    # Extract diagonal & off-diagonal
    diag = torch.diag(torch.diag(mm))
    off_diag = mm - diag

    diag = diag.sum()
    off_diag = off_diag.sum()

    return diag, off_diag, diag + off_diag * lmd

class _Model(nn.Module):

    def __init__(self, model: nn.Module, transforms: List):
        super().__init__()

        self.model = model
        self.trfms = nn.Sequential(*transforms)

    def forward(self, x):
        xx = [x, self.trfms(x)]
        for i in range(2):
            _x = xx[i]
            _x = self.model(_x)
            _x = (_x - _x.mean(0))/_x.std(0)
            xx[i] = _x

        return xx

class _LitModel(LightningModule):

    def __init__(self, model: nn.Module, lmd: float, opt: str, opt_kwargs: Dict[str, float]):
        super().__init__()

        self.model = model
        self.lmd = lmd

        self.opt = _opts.get(opt, AdamW)
        self.opt_kwargs = opt_kwargs

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = self.opt(self.parameters(), **self.opt_kwargs)
        return opt

    def training_step(self, x, *args):
        # TODO: make sure the batch is a tensor with no labels
        # x, *_ = x
        z1, z2 = self(x)
        diag, off_diag, loss = _loss(z1, z2, self.lmd)

        self.log('total_loss', loss, prog_bar=True)
        self.log('diag_loss', diag, prog_bar=True)
        self.log('off_diag_loss', off_diag, prog_bar=True)

        return loss


def barlow_twin(model: nn.Module, transforms: List, dataloader: Union[DataLoader,LightningDataModule], lmd: float=.05, opt:str='AdamW', opt_kwargs: Dict[str, float]={'lr':1e-3},  **trainer_kwargs):
    """ This function trains the model following the Barlow Twin method to reduce redundancy in components.
        
    During training, this function shows three metrics in the progress bar, namely: total loss, diagonal loss, off diagonal loss.
    
    Note:
        For more, check the original paper: https://arxiv.org/abs/2103.03230

    Args:
        model (nn.Module): PyTorch module that returns an Matrix output whose shape is [batch, D]; D could be any integer larger than 0.
        transforms : List of torchvision scriptible transforms (working with tensors but not with PIL images). It's used to generate one distorted version 
                    of the original image, the other version is the orginal image itself.
        dataloader: A PyTorch DataLoader or a Pytorch-Lightning DataModule. 
                        The retrieved batch at each step must contain only the input without any additional tensors.
        lmd (float. Optional ): The lambda parameter in the original paper.
                                    Default is 0.05.
        opt (str, Optional): String referring to the desired optimizer to be used. 
                                These are the supported ones: [Adam, AdamW, SGD, RMSprop, Adagrad], 
                                Default is 'AdamW'.
        opt_kwargs (Dict): Dictionary of optimizer configuration. 
                            Default is {'lr':1e-3}.
        trainer_kwargs: Sequence of Pytorch-Lightning Trainer configuration. For e.g: max_epochs=8, gpus=-1, ...

    Returns:
        The resulting pre-trained model.
    """


    model = _Model(model, transforms=transforms)
    litmodel = _LitModel(model, lmd, opt, opt_kwargs)
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(litmodel, dataloader)

    return model.model
