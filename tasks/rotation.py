from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, Dict
from collections import OrderedDict

import torch
import numpy as np

from torch import Tensor, nn
from torch.nn import functional as F

from .bases import AuxiliaryTask
from common.losses import LossInfo
from common.layers import Flatten


def rotate(x: Tensor, angle: int) -> Tensor:
    # TODO: Test that this works.
    assert angle % 90 == 0, "can only rotate 0, 90, 180, or 270 degrees"
    k = angle // 90
    if angle == 0:
        return x
    rot_x = np.ascontiguousarray(np.rot90(x, k, axes=(2,3)))
    rot_x_tensor = torch.from_numpy(rot_x)
    return rot_x_tensor
    raise NotImplementedError("TODO")


class RotationTask(AuxiliaryTask):
    def __init__(self, options: AuxiliaryTask.Options=None):
        super().__init__(options)
        self.classify_rotation: nn.Linear = nn.Sequential(
            Flatten(),
            nn.Linear(AuxiliaryTask.hidden_size, 4),
        )
        self.loss = nn.CrossEntropyLoss()

    def get_loss(self,
                 x: Tensor,
                 h_x: Tensor,
                 y_pred: Tensor,
                 y: Tensor=None) -> LossInfo:
        batch_size = x.shape[0]
        # TODO: change AuxiliaryTask so it also returns a dict with each loss?
        loss_info = LossInfo()
        
        # no rotation:
        rot_label = torch.zeros([batch_size], dtype=torch.long)
        rot_pred = self.classify_rotation(h_x)
        rot_loss = self.loss(rot_pred, rot_label)
        
        loss_info.losses["rotation_0"] = rot_loss
        loss_info.total_loss += rot_loss

        for rotation_degrees in [90, 180, 270]:
            rot_x = rotate(x, rotation_degrees)
            rot_label = torch.ones([batch_size], dtype=torch.long) * 1
            rot_h_x = self.encode(rot_x)
            rot_pred = self.classify_rotation(rot_h_x)
            rot_loss = self.loss(rot_pred, rot_label)
            
            loss_info.losses[f"rotation_{rotation_degrees}"] = rot_loss
            loss_info.total_loss += rot_loss
        
        return loss_info