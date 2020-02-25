from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.bases import AuxiliaryTask


def mixup(x1: Tensor, x2: Tensor, coeff: Tensor) -> Tensor:
    return x1 + (x2 - x1) * coeff


class Mixup(AuxiliaryTask):
    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tensor:
        batch_size = x.shape[0]
        coeff = torch.rand(batch_size)
        
        x1 = x[:batch_size//2]
        x2 = x[batch_size//2:]
        mix_x = mixup(x1, x2, coeff)
        
        y_pred_1 = y_pred[:batch_size//2]
        y_pred_2 = y_pred[batch_size//2:]
        y_pred_mix = mixup(y_pred_1, y_pred_2, coeff)

        mix_h_x = self.encoder(mix_x)
        mix_y_pred = self.classifier(mix_h_x)

        difference = y_pred_mix - mix_y_pred
        return (difference **2).sum()


class ManifoldMixup(AuxiliaryTask):
    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tensor:
        batch_size = x.shape[0]
        coeff = torch.rand(size=batch_size)
        
        h1 = h_x[:batch_size//2]
        h2 = h_x[batch_size//2:]
        mix_h_x = mixup(h1, h2, coeff)
        
        y_pred_1 = y_pred[:batch_size//2]
        y_pred_2 = y_pred[batch_size//2:]
        y_pred_mix = mixup(y_pred_1, y_pred_2, coeff)

        mix_y_pred = self.classifier(mix_h_x)

        difference = y_pred_mix - mix_y_pred
        return (difference **2).sum()
