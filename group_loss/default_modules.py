import torch
import torch.nn as nn
from .base_classes import HierarchicalGroupWrapper
from typing import List


class HierarchicalConv2d(HierarchicalGroupWrapper):
    """Сверточный слой с группировкой параметров"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 param_groups: List[str], **kwargs):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        super().__init__(conv, param_groups)

    def get_weights(self):
        weight = self.module.weight  # [out_c, in_c, k, k]
        # Flatten spatial & channel dims per filter
        weights = weight.view(weight.size(0), -1)
        return weights


class HierarchicalLinear(HierarchicalGroupWrapper):
    """Линейный слой с группировкой параметров"""
    def __init__(self, in_features: int, out_features: int, 
                 param_groups: List[str], **kwargs):
        linear = nn.Linear(in_features, out_features, **kwargs)
        super().__init__(linear, param_groups)

    def get_weights(self):
        weight = self.module.weight
        # Flatten spatial & channel dims per filter
        weights = weight.view(weight.size(0), -1)
        return weights