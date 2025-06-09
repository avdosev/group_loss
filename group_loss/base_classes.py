import torch
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

    def count_zero_groups(self, threshold: float = 1e-5) -> Dict[str, int]:
        """Return counts of zero and near-zero base-level groups.

        A group is considered **zero** if *all* its weights equal zero.
        It is **near-zero** if the maximum absolute value within the group is
        below ``threshold`` while not being exactly zero.  This heuristic makes
        the criterion independent of the number of weights in a group and thus
        comparable across convolutional and linear layers.
        """
        w = self.get_weights().detach().cpu()
        abs_max = w.abs().max(dim=1).values
        total = abs_max.numel()
        zeros = (abs_max == 0).sum().item()
        near_zeros = ((abs_max < threshold) & (abs_max != 0)).sum().item()
        return {"total": total, "zeros": zeros, "near_zeros": near_zeros}

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


def hierarchical_zero_statistics(
    model: HierarchicalModule, threshold: float = 1e-5
) -> Dict[str, Dict[str, int]]:
    """Return zero/near-zero counts for each hierarchy tag in *model*.

    Parameters are aggregated according to :class:`HierarchicalGroup` tags.  For
    the special tag ``"base_level"`` the counts are computed per filter/neurone
    using :meth:`HierarchicalGroupWrapper.count_zero_groups`.  For all other
    tags a group is considered **zero** when **all** its parameters are exactly
    zero and **near-zero** when the maximum absolute value is below ``threshold``
    while not being exactly zero.
    """

    stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "zeros": 0, "near_zeros": 0}
    )

    def _gather_weights(grp: HierarchicalGroup) -> torch.Tensor:
        tensors = []
        if isinstance(grp.module, HierarchicalGroupWrapper):
            tensors.append(grp.module.get_weights().detach().cpu().view(-1))
        elif isinstance(grp.module, list):
            for item in grp.module:
                if isinstance(item, HierarchicalGroupWrapper):
                    tensors.append(item.get_weights().detach().cpu().view(-1))
                elif isinstance(item, HierarchicalGroup):
                    tensors.append(_gather_weights(item))
        return torch.cat(tensors) if tensors else torch.tensor([])

    def _traverse(grp: HierarchicalGroup) -> None:
        if isinstance(grp.module, HierarchicalGroupWrapper):
            unit_stats = grp.module.count_zero_groups(threshold)
            for tag in grp.groups:
                if tag == "base_level":
                    stats[tag]["total"] += unit_stats["total"]
                    stats[tag]["zeros"] += unit_stats["zeros"]
                    stats[tag]["near_zeros"] += unit_stats["near_zeros"]
                else:
                    w = _gather_weights(grp)
                    if w.numel():
                        is_zero = int(torch.all(w == 0).item())
                        is_near = int(w.abs().max().item() < threshold and not is_zero)
                        stats[tag]["total"] += 1
                        stats[tag]["zeros"] += is_zero
                        stats[tag]["near_zeros"] += is_near
        else:  # grp.module is a list of sub-groups/modules
            w = _gather_weights(grp)
            if w.numel():
                is_zero = int(torch.all(w == 0).item())
                is_near = int(w.abs().max().item() < threshold and not is_zero)
                for tag in grp.groups:
                    stats[tag]["total"] += 1
                    stats[tag]["zeros"] += is_zero
                    stats[tag]["near_zeros"] += is_near
            if isinstance(grp.module, list):
                for item in grp.module:
                    if isinstance(item, HierarchicalGroup):
                        _traverse(item)
                    elif isinstance(item, HierarchicalGroupWrapper):
                        _traverse(item.get_param_groups())

    for root in model.get_param_groups():
        _traverse(root)

    return stats
    