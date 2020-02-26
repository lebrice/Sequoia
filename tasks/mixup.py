from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bases import AuxiliaryTask


def mixup(x1: Tensor, x2: Tensor, coeff: Tensor) -> Tensor:
    assert coeff.dim() == 1
    assert x1.shape == x2.shape
    n = x1.shape[0]
    assert n == coeff.shape[0], coeff.shape
    shape = [n]
    shape.extend([1 for _ in x1.shape[1:]])
    coeff = coeff.view(shape)
    coeff = coeff.expand_as(x1)
    # return x1 + (x2 - x1) * coeff    
    return torch.lerp(x1, x2, coeff)

        

class MixupTask(AuxiliaryTask):
    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tensor:
        batch_size = x.shape[0]
        assert batch_size % 2  == 0, "Can only mix an even number of samples."
        
        mix_coeff = torch.rand(batch_size//2)

        x1 = x[0::2]
        x2 = x[1::2]
        mix_x = mixup(x1, x2, mix_coeff)
        
        y_pred_1 = y_pred[0::2]
        y_pred_2 = y_pred[1::2]
        y_pred_mix = mixup(y_pred_1, y_pred_2, mix_coeff)
        
        if self.preprocessing:
            mix_x = self.preprocessing(mix_x)
            
        mix_h_x = self.encoder(mix_x)
        mix_y_pred = self.classifier(mix_h_x)

        difference = y_pred_mix - mix_y_pred
        return (difference **2).sum()


class ManifoldMixupTask(AuxiliaryTask):
    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tensor:
        batch_size = x.shape[0]
        assert batch_size % 2  == 0, "Can only mix an even number of samples."
        mix_coeff = torch.rand(batch_size//2)

        h1 = h_x[0::2]
        h2 = h_x[1::2]
        mix_h_x = mixup(h1, h2, mix_coeff)
        
        y_pred_1 = y_pred[0::2]
        y_pred_2 = y_pred[1::2]
        y_pred_mix = mixup(y_pred_1, y_pred_2, mix_coeff)

        mix_y_pred = self.classifier(mix_h_x)

        difference = y_pred_mix - mix_y_pred
        return (difference **2).sum()
