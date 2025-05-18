import torch
import torch.nn as nn
import torch.nn.functional as F

from group_loss.base_classes import HierarchicalModule
from group_loss.default_modules import HierarchicalConv2d, HierarchicalLinear

from typing import List


class HierarchicalResBlock(HierarchicalModule):
    """Residual блок с иерархической группировкой"""
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = HierarchicalConv2d(
            in_planes, planes, kernel_size=3,
            param_groups=['level1', 'block'],
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = HierarchicalConv2d(
            planes, planes, kernel_size=3,
            param_groups=['level1', 'block'],
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                HierarchicalConv2d(
                    in_planes, planes, kernel_size=1,
                    param_groups=['level1', 'block'],
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class HierarchicalResNet(HierarchicalModule):
    """Полная реализация ResNet с иерархической группировкой"""
    def __init__(self, num_blocks: List[int], num_classes: int = 10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = HierarchicalConv2d(3, 64, kernel_size=3,
                                      param_groups=['level1', 'layer'],
                                      stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = HierarchicalLinear(512, num_classes, param_groups=['layer'])

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(HierarchicalResBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(HierarchicalResBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


#########################################


def get_resnet18(num_classes: int = 1000) -> HierarchicalResNet:
    """Return ResNet-18 with hierarchical metadata."""
    return HierarchicalResNet(num_blocks=[2, 2, 2, 2], num_classes=num_classes)


def get_resnet34(num_classes: int = 1000) -> HierarchicalResNet:
    """Return ResNet-34 with hierarchical metadata."""
    return HierarchicalResNet(num_blocks=[3, 4, 6, 3], num_classes=num_classes)
