import torch.nn as nn
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List


class HierarchicalGroup():
    def __init__(self, wrapper, groups: List[str], name=''):
        self.module = wrapper
        self.groups = groups
        self.name = name

    def dict(self):
        return {
            'module': self.module if not isinstance(self.module, list) else [item.dict() if isinstance(item, HierarchicalGroup) else item for item in self.module],
            'group': self.groups,
            'name': self.name,
        }
    
    def __str__(self):
        return str(self.dict())
    

class HierarchicalGroupWrapper(nn.Module):
    """Обертка для добавления мета-информации о группировке параметров"""
    def __init__(self, module: nn.Module, groups: List[str]):
        super().__init__()
        self.module = module
        self._param_groups = groups
        
    def get_param_groups(self) -> HierarchicalGroup:
        return HierarchicalGroup(
            self, self._param_groups
        )
    
    @abstractmethod
    def get_weights(self):
        pass
    
    def forward(self, x):
        return self.module.forward(x)


class HierarchicalModule(nn.Module):
    """Базовый класс для моделей с иерархической группировкой параметров"""
    
    def get_param_groups(self) -> List[HierarchicalGroup]:
        groups = []        
        for name, module in self.named_children():
            groups.extend(extract_groups(module))
        return groups


def extract_groups(module: nn.Module):
    groups = []
    if isinstance(module, HierarchicalGroupWrapper):
        groups.append(module.get_param_groups())
    if isinstance(module, HierarchicalModule):
        groups.extend(module.get_param_groups())
    if isinstance(module, nn.Sequential):
        for child in module.children():
            groups.extend(extract_groups(child))
    return groups


def make_hierarchy(blocks: list, group_level, name=''):
    if isinstance(group_level, str):
        group_level = [group_level]
    return [HierarchicalGroup(blocks, group_level, name)]
    