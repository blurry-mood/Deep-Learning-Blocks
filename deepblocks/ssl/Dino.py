from copy import deepcopy
from typing import Dict, Tuple, Union

from pytorch_lightning import LightningModule, Trainer, LightningDataModule

from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad
from torchvision import transforms

_opts = {'Adam': Adam, 'AdamW': AdamW, 'SGD': SGD,
         'RMSprop': RMSprop, 'Adagrad': Adagrad}

_tps = .1
_tpt = .1
_l = .96
_m = .96
_D = 0

@torch.jit.script
def _loss(t: torch.Tensor, s: torch.Tensor):
    # t: N x 2 x D 
    # s: N x (M + 2) x D
    l = - (t[:, :1] * s[:, 1:]).sum(dim=2).mean()
    l += - (t[:, 1:] * s[:, 2:]).sum(dim=2).mean()
    l += - (t[:, 1:] * s[:, :1]).sum(dim=2).mean()

    return l

class _Cropper(nn.Module):

    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        super().__init__()

        self.local_crops_number = local_crops_number
        _flip_and_color_jitter = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ]
        _normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # first global crop
        global_transfo1 = [
            transforms.RandomResizedCrop(224, scale=global_crops_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            *_flip_and_color_jitter,
            transforms.GaussianBlur(5),
            _normalize
        ]
        
        # second global crop
        global_transfo2 = [
            transforms.RandomResizedCrop(224, scale=global_crops_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            *_flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.1),
            transforms.RandomSolarize(128, p=0.2),
            _normalize
        ]
        
        # Local crops
        local_transfo = [
             transforms.RandomResizedCrop(96, scale=local_crops_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            *_flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
            _normalize
        ]
        
        # first global crop
        self.global_transfo1 = nn.Sequential(
            *global_transfo1
        )

        # second global crop
        self.global_transfo2 = nn.Sequential(
            *global_transfo2
        )

        
        self.local_transfo = nn.Sequential(
           *local_transfo
        )

    def forward(self, x):
        crops = [self.global_transfo1(x), self.global_transfo2(x)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(x))
        return crops


class _LitModel(LightningModule):

    def __init__(self, student, teacher, cropper, opt: str, opt_kwargs: Dict[str, float]):
        super().__init__()

        self.student = student
        self.teacher = teacher
        # there is no backpropagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # Make it a parameter to avoid cpu-gpu operation issue.
        self.C = nn.Parameter(torch.zeros(_D), requires_grad=False)

        self.cropper = cropper

        self.opt = _opts.get(opt, AdamW)
        self.opt_kwargs = opt_kwargs

    def forward(self, imgs):
        crops = self.cropper(imgs)

        # Teacher & student crops
        t_crops, s_crops = [], []
        for i in range(2):
            t_crops.append(self.teacher(crops[i]).unsqueeze(1))
        for crop in crops:
            s_crops.append(self.student(crop).unsqueeze(1))

        return torch.cat(t_crops, dim=1), torch.cat(s_crops, dim=1)

    def configure_optimizers(self):
        opt = self.opt(self.parameters(), **self.opt_kwargs)
        return opt

    def training_step(self, x, *args):      
        t, s = self(x)
        t, s = self.softmax((t - self.C) / _tpt), self.logsoftmax(s / _tps)
        loss = _loss(t, s)
        return {'loss': loss, 't': t}

    def training_step_end(self, batch_parts):
        t = batch_parts['t']  # N x 2 x D
        t = torch.flatten(t, 0, 1) # 2N x D

        # Update teacher params
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data = _l * t.data + (1 - _l) * s.data

        # Update value of mean C
        self.C.data = _m * self.C.data + (1 - _m) * t.mean(dim=0)
        
        return batch_parts['loss']


def dino(model: nn.Module, output_dim: int, dataloader: Union[LightningDataModule, DataLoader],
         global_crops_scale: Tuple[float, float], local_crops_scale: Tuple[float, float],
          local_crops_number: int,
         tps: float=.9, tpt: float=.9, l: float=.9, m: float=.9,
         opt: str = 'AdamW', opt_kwargs: Dict = {'lr': 1e-4}, **trainer_kwargs
         ):
    """ Pre-train a model in a self-supervised fashion using DINO framework (Knowledge DIstillation with NO labels).

    The function clones the given model twice yielding the student & teacher networks, the passed model remains intact.
    
    Note:
        To get better grasp on this framework, check the original paper: https://arxiv.org/abs/2104.14294 
    
    Args:
        model: PyTorch module whose output is a 2D tensor: [batch, output_dim]. 
                The model should return a fixed-sized tensor if given images with different sizes: 224 & 96.
        output_dim: Integer denoting the last dimension of the model output. 
        dataloader: A PyTorch DataLoader or a Pytorch-Lightning DataModule. 
                        The retrieved batch, at each step, must contain only the input without any additional tensors.
        global_crops_scale: Tuple with two floats in [0, 1], used to crop large patches of the image using torchvision.transforms.RandomResizedCrop .
        local_crops_scale: Tuple with two floats in [0, 1], used to crop small patches of image using torchvision.transforms.RandomResizedCrop.
        local_crops_number: Integer denoting the number of small patches to use for the student network.
        tps, tpt: Floats denoting the temperature used to scale student and teacher outputs respectively.
        l: Float used when updating the teacher parameters using the formula: t = l * t + (1 - l) * s.
        m: Float used when updating the value of the mean C.
        opt: String referring to the desired optimizer to be used. 
                These are the supported ones: [Adam, AdamW, SGD, RMSprop, Adagrad], 
                otherwise, AdamW is used by default.
        opt_kwargs: Dictionary of optimizer configuration.
        trainer_kwargs: Sequence of Pytorch-Lightning Trainer configuration. For e.g: max_epochs=8, gpus=-1, ...

    Returns:
        The pre-trained student model.
    """
    global _D, _tps, _tpt, _l, _m

    _D = output_dim
    _tps = tps
    _tpt = tpt
    _l = l
    _m = m

    student = deepcopy(model)
    teacher = deepcopy(model)
    # del model

    litmodel = _LitModel(student, teacher,
                         _Cropper(global_crops_scale,
                                  local_crops_scale, local_crops_number),
                         opt, opt_kwargs)

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(litmodel, dataloader)

    return litmodel.student