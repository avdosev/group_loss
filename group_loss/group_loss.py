import torch
from typing import List
from .base_classes import *

#########################################


class HierarchicalRegularizer:
    def __init__(self, model: HierarchicalModule, hierarchy_config):
        """
        hierarchy_config: список словарей с параметрами для каждого уровня
        Пример: [{'level': 0, 'lambda': 0.01, 'norm': 'L1'}, ...]
        """
        self.hierarchy = hierarchy_config
        self.parameters = self._extract_parameters(model)
        
    def _extract_parameters(self, model: HierarchicalModule):
        return model.get_parameter_groups()

    def compute_loss(self):
        total_loss = 0
        for config in self.hierarchy:
            level = config['level']
            lam = config['lambda']
            norm = config['norm']
            
            if level not in self.parameters:
                continue
                
            for p in self.parameters[level]:
                if norm == 'L1':
                    total_loss += lam * torch.norm(p, p=1)
                elif norm == 'L2':
                    total_loss += lam * torch.norm(p, p=2)
        return total_loss
    
    def compute_nested_loss(self):
        total_loss = 0
        hierarchy_tree = {
            0: {'children': [1], 'norm': 'L2', 'lambda': 0.01},
            1: {'children': [2], 'norm': 'L1', 'lambda': 0.001},
            2: {'children': [], 'norm': 'L1', 'lambda': 0.0001}
        }
        
        def compute_level(level):
            if not hierarchy_tree[level]['children']:
                # Базовый случай: листовой уровень
                params = self.parameters.get(level, [])
                norms = [torch.norm(p, p=hierarchy_tree[level]['norm'][1:]) for p in params]
                return sum(norms) * hierarchy_tree[level]['lambda']
            else:
                # Рекурсивный случай
                child_norms = [compute_level(child) for child in hierarchy_tree[level]['children']]
                combined_norm = torch.norm(torch.stack(child_norms), p=hierarchy_tree[level]['norm'][1:])
                return combined_norm * hierarchy_tree[level]['lambda']
        
        return compute_level(0)

    def compute_loss(self, config=None):
            if config is None:
                config = self.config
                
            loss = 0
            
            if config['type'] == 'global':
                for param in self.model.parameters():
                    if config['norm'] == 'L1':
                        loss += config['lambda'] * torch.norm(param, p=1)
                    elif config['norm'] == 'L2':
                        loss += config['lambda'] * torch.norm(param, p=2)

            elif config['type'] == 'layerwise':
                for group in self.param_groups.get(config['groups'], []):
                    group_loss = 0
                    for param in group:
                        if config['norm'] == 'L1':
                            group_loss += torch.norm(param, p=1)
                        elif config['norm'] == 'L2':
                            group_loss += torch.norm(param, p=2)
                    loss += config['lambda'] * group_loss

            elif config['type'] == 'hierarchical':
                child_losses = []
                for child_config in config['children']:
                    child_loss = self.compute_loss(child_config)
                    child_losses.append(child_loss)
                
                combined = torch.stack(child_losses)
                loss = config['lambda'] * torch.norm(combined, p=2 if config['norm']=='L2' else 1)

            return loss

# === Hierarchical L1/L2 Loss ===
def hierarchical_group_l1(params: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack([
        p.abs().sum(dim=(1, 2, 3)).sum() for p in params if len(p.shape) == 4
    ]).sum()

def hierarchical_group_l2(params: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack([
        (p ** 2).sum(dim=(1, 2, 3)).sqrt().sum() for p in params if len(p.shape) == 4
    ]).sum()

def hierarchical_nested_l1(groups: List[List[torch.Tensor]]) -> torch.Tensor:
    return torch.stack([
        torch.stack([p.abs().sum() for p in group]).sum() for group in groups
    ]).sum()

def hierarchical_nested_l2(groups: List[List[torch.Tensor]]) -> torch.Tensor:
    return torch.stack([
        torch.stack([(p**2).sum().sqrt() for p in group]).sum() for group in groups
    ]).sum()
