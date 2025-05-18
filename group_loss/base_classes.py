import torch.nn as nn
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List


class HierarchicalGroupWrapper(nn.Module):
    """Обертка для добавления мета-информации о группировке параметров"""
    def __init__(self, module: nn.Module, groups: List[str]):
        super().__init__()
        self.module = module
        self._param_groups = groups
        
    def get_param_groups(self):
        return self._param_groups
    
    @abstractmethod
    def get_weights(self):
        pass
    
    def forward(self, x):
        return self.module.forward(x)


class HierarchicalModule(nn.Module):
    """Абстрактный базовый класс для моделей с иерархической группировкой параметров"""
    
    def get_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        groups = defaultdict(list)
        for name, module in self.named_children():
            if isinstance(module, HierarchicalGroupWrapper):
                for group in module.get_param_groups():
                    groups[group].extend(module.parameters())
        return groups
