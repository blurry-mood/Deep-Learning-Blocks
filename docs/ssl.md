
# **Barlow Twin**
A self-supervised learning method with the aim of reducing the redundancy in the network's ouput layer using the cross-correlation matrix.   
For more, check the original paper: https://arxiv.org/abs/2103.03230

### **How to use**:
1- Import Barlow Twin function:
> from deepblocks.ssl import barlow_twin

2- Train:
>  pretrained = barlow_twin(model: nn.Module, transforms: List, dataloader: Union[DataLoader,LightningDataModule], lmd: float=.05, opt:str='AdamW', opt_kwargs: Dict[str, float]={'lr':1e-3},  **trainer_kwargs)  
- `model`: PyTorch module that returns an Matrix output whose shape is [batch, D]; D could be any integer larger than 0.
- `transforms`: List of torchvision scriptible transforms (working with tensors not PIL images). It's used to generate one distorted version of the original image, the other version is the orginal image itself.
- `dataloader`: A PyTorch DataLoader or a Pytorch-Lightning DataModule. 
The retrieved batch at each step must contain only the input without any additional tensors.
- `lmd`: The lambda parameter in the original paper.
- `opt`: String referring to the desired optimizer to be used. 
These are the supported ones: [Adam, AdamW, SGD, RMSprop, Adagrad], 
otherwise, AdamW is used by default or in case of invalid value.
- `opt_kwargs`: Dictionary of optimizer configuration.
- `trainer_kwargs`: Sequence of Pytorch-Lightning Trainer configuration. For e.g: max_epochs=8, gpus=-1, ...
