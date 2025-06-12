import torch
import pytest

from group_loss.group_loss import HierarchicalRegularizer
from group_loss.default_modules import HierarchicalConv2d, HierarchicalLinear
from group_loss.base_classes import HierarchicalModule


class ToyModel(HierarchicalModule):
    def __init__(self):
        super().__init__()
        self.conv = HierarchicalConv2d(1, 2, kernel_size=1, param_groups=["base_level"])
        self.fc = HierarchicalLinear(4, 2, param_groups=["base_level"])
        torch.nn.init.constant_(self.conv.module.weight, 1.0)
        torch.nn.init.constant_(self.fc.module.weight, 1.0)


def _loss(cfg):
    model = ToyModel()
    reg = HierarchicalRegularizer(cfg)
    return float(reg.forward(model))


def test_l0approx():
    cfg = {
        "type": "layerwise",
        "groups": "base_level",
        "norm": "L1",
        "inner_norm": "L0",
        "lambda": 1.0,
    }
    expected = 2 * 1 + 2 * (4**0.1)
    assert pytest.approx(_loss(cfg), rel=1e-4) == expected
