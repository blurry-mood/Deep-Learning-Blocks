""" Pretrain PyTorch models using either:
    - the `DINO` paradigm, 
    - or the `Barlow Twin` method.

For both, **deepblocks** leverages **PyTorch-Lightning** package to perform the training.
"""
from .BarlowTwin import barlow_twin
from .Dino import dino