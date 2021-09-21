""" **Loss** implements loss functions.

It contains:
    - KLDivLoss: computes the KL-divergence loss,
    - FocalLoss: for both binary & multi-class settings,
    - AUCLoss & AUCMarginLoss.
"""

from .FocalLoss import FocalLoss
from .KLDivLoss import KLDivLoss
from .AUCLoss import AUCLoss, AUCMarginLoss
