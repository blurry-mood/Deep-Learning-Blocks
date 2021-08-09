
# **Barlow Twin**
A self-supervised learning method with the aim of reducing the redundancy in the network's ouput layer using the cross-correlation matrix.   
For more, check the original paper: https://arxiv.org/abs/2103.03230

### **How to use**:
1- Import Barlow Twin function:
> from deepblocks.ssl import barlow_twin

2- Pretrain:
>  pretrained = barlow_twin(model: nn.Module, transforms: List, dataloader: Union[DataLoader,LightningDataModule], lmd: float=.05, opt:str='AdamW', opt_kwargs: Dict[str, float]={'lr':1e-3},  **trainer_kwargs)  
* `model`: PyTorch module that returns an Matrix output whose shape is [batch, D]; D could be any integer larger than 0.
* `transforms`: List of torchvision scriptible transforms (working with tensors not PIL images). It's used to generate one distorted version of the original image, the other version is the orginal image itself.
* `dataloader`: A PyTorch DataLoader or a Pytorch-Lightning DataModule. 
The retrieved batch at each step must contain only the input without any additional tensors.
* `lmd`: The lambda parameter in the original paper.
* `opt`: String referring to the desired optimizer to be used. 
These are the supported ones: [Adam, AdamW, SGD, RMSprop, Adagrad], 
otherwise, AdamW is used by default or in case of invalid value.
* `opt_kwargs`: Dictionary of optimizer configuration.
* `trainer_kwargs`: Sequence of Pytorch-Lightning Trainer configuration. For e.g: max_epochs=8, gpus=-1, ...


# **DINO**:
Check the original paper: https://arxiv.org/abs/2104.14294 

### **How to use**:
1- Import DINO function:
> from deepblocks.ssl import dino

2- Pretrain:
>  pretrained = dino(model: nn.Module, output_dim: int, dataloader: Union[LightningDataModule, DataLoader], global_crops_scale: Tuple[float, float], local_crops_scale: Tuple[float, float], local_crops_number: int, tps: float=.9, tpt: float=.9, l: float=.9, m: float=.9, opt: str = 'AdamW', opt_kwargs: Dict = {'lr': 1e-4}, **trainer_kwargs)

* `model`: PyTorch module whose output is a 2D tensor: [batch, output_dim]. The model should return a fixed-sized tensor if given images with different sizes: 224 & 96.
* `output_dim`: Integer denoting the last dimension of the model output. 
* `dataloader`: A PyTorch DataLoader or a Pytorch-Lightning DataModule. 
The retrieved batch, at each step, must contain only the input without any additional tensors.
* `global_crops_scale`: Tuple with two floats in [0, 1], used to crop large patches of the image using torchvision.transforms.RandomResizedCrop .
* `local_crops_scale`: Tuple with two floats in [0, 1], used to crop small patches of image using torchvision.transforms.RandomResizedCrop.
* local_crops_number: Integer denoting the number of small patches to use for the student network.
* `tps, tpt`: Floats denoting the temperature used to scale student and teacher outputs respectively.
* `l`: Float used when updating the teacher parameters using the formula: t = l * t + (1 - l) * s.
* `m`: Float used when updating the value of the mean C.
* `opt`: String referring to the desired optimizer to be used. These are the supported ones: [Adam, AdamW, SGD, RMSprop, Adagrad], 
otherwise, AdamW is used by default.
- `opt_kwargs`: Dictionary of optimizer configuration.
* `trainer_kwargs`: Sequence of Pytorch-Lightning Trainer configuration. For e.g: max_epochs=8, gpus=-1, ...
* `pretrained`: The pretrained student model.