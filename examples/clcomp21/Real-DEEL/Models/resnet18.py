from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box
from .model import Model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Resnet18(Model):
    def __init__(self, image_space: Box, n_classes: int, bic: bool = False) -> None:
        # assertion for Conv networks
        assert len(image_space.shape) == 3
        modules_list, penulimate_layer_indx = self._get_modules(image_space, n_classes)
        super(Resnet18, self).__init__(modules_list, penulimate_layer_indx, n_classes, bic)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _get_modules(self, image_space: Box, n_classes: int):
        self.in_planes = 64
        block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        modules_list = []

        modules_list.append(
            nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv1",
                            nn.Conv2d(
                                3, 64, kernel_size=3, stride=1, padding=1, bias=False
                            ),
                        ),
                        ("bn1", nn.BatchNorm2d(64)),
                        (
                            "layer1",
                            self._make_layer(block, 64, num_blocks[0], stride=1),
                        ),
                    ]
                )
            )
        )
        modules_list.append(self._make_layer(block, 128, num_blocks[1], stride=2))
        modules_list.append(self._make_layer(block, 256, num_blocks[2], stride=2))
        modules_list.append(self._make_layer(block, 512, num_blocks[3], stride=2))
        penulimate_layer_indx = len(modules_list)
        modules_list.append(
            nn.Sequential(
                OrderedDict(
                    [
                        ("avgPool", nn.AvgPool2d(kernel_size=4)),
                        ("flatten", nn.Flatten()),
                        ("linear", nn.Linear(512 * block.expansion, n_classes)),
                    ]
                )
            )
        )
        return modules_list, penulimate_layer_indx
