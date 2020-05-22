from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, Set, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from collections import OrderedDict
from common.losses import LossInfo

from .auxiliary_task import AuxiliaryTask
import logging

logger = logging.getLogger(__file__)

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


from copy import deepcopy
from utils.utils import add_dicts


def average_models(old: nn.Module, new: nn.Module, old_frac: float = 0.1) -> None:
    """ Updates the old model weights with an exponential moving average of the new.
    
    If the weight is present in both models, the new weight value will be
    `v = old_frac * old_value + (1-old_frac) * new_value`
    otherwise, if the weight is only present in either, keeps the value as-is.

    Returns nothing, as it modifies the `old` module in-place.   
    """
    old_state = old.state_dict()
    new_state = new.state_dict()

    all_keys: Set[str] = set(old_state.keys()).union(set(new_state.keys()))

    result: Dict[str, Tensor] = OrderedDict()
    for k in all_keys:
        v_old = old_state.get(k)
        v_new = new_state.get(k)
        if v_old is not None and v_new is not None:
            v = old_frac * v_old + (1 - old_frac) * v_new
        elif v_old is not None:
            v = v_old
        elif v_new is not None:
            v = v_new
        result[k] = v
    missing, unexpected = old.load_state_dict(result, strict=False)
    if missing:
        logger.debug(f"Missing keys: {missing}")
    if unexpected:
        logger.debug(f"Unexpected keys: {unexpected}")


class MixupTask(AuxiliaryTask):

    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Options for the Mixup (ICT) Task. """
        # Fraction of the old weights to use in the mean-teacher model. 
        mean_teacher_mixing_coefficient: float = 0.9 

    def __init__(self,
                 coefficient: float=None,
                 name: str="Mixup",
                 options: "MixupTask.Options"=None):
        super().__init__(coefficient=coefficient, name=name, options=options)
        self.options: MixupTask.Options
        # TODO: Add the mean-teacher model to the ICT.
        self.mean_encoder = deepcopy(AuxiliaryTask.encoder)
        self.mean_classifier = deepcopy(AuxiliaryTask.classifier)
    
    def mean_encode(self, x: Tensor) -> Tensor:
        x = AuxiliaryTask.preprocessing(x)
        return self.mean_encoder(x)

    def mean_logits(self, h_x: Tensor) -> Tensor:
        return self.mean_classifier(h_x)

    def on_model_changed(self, global_step: int)-> None:
        """ Executed when the model was updated. """
        average_models(
            self.mean_encoder,
            AuxiliaryTask.encoder,
            old_frac=self.options.mean_teacher_mixing_coefficient
        )
        average_models(
            self.mean_classifier,
            AuxiliaryTask.classifier,
            old_frac=self.options.mean_teacher_mixing_coefficient
        )


    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> LossInfo:
        #select only unlabelled examples like in ICT: https://arxiv.org/pdf/1903.03825.pdf
        x = x[len(y):]
        h_x = h_x[len(y):]
        batch_size = x.shape[0]
        y_pred = y_pred[len(y):]

        # assert batch_size % 2  == 0, f"Can only mix an even number of samples. (batch size is {batch_size})"
        if batch_size % 2 != 0:
            x = x[:-1]
            y_pred = y_pred[:-1]

        from .tasks import Tasks
        loss_info = LossInfo(name=Tasks.MIXUP)
        if batch_size > 0:
            mix_coeff = torch.rand(batch_size//2, dtype=x.dtype, device=x.device)

            x1 = x[0::2]
            x2 = x[1::2]

            mix_x = mixup(x1, x2, mix_coeff)
            loss_info.tensors["mix_x"] = mix_x.detach()
            mix_h_x = self.encode(mix_x)
            mix_y_pred = self.classifier(mix_h_x)

            # Use the mean teacher to get the h_x and y_pred for the unlabeled data.
            h_x = self.mean_encode(x)
            y_pred = self.mean_logits(h_x)
            y_pred_1 = y_pred[0::2]
            y_pred_2 = y_pred[1::2]
            y_pred_mix = mixup(y_pred_1, y_pred_2, mix_coeff)
            loss_info.tensors["y_pred_mix"] = y_pred_mix.detach()

            loss = torch.dist(y_pred_mix, mix_y_pred)
            loss_info.total_loss = loss
        else:
            loss_info.total_loss = torch.tensor([0])

        return loss_info


class ManifoldMixupTask(AuxiliaryTask):
    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> LossInfo:
        batch_size = x.shape[0]
        # assert batch_size % 2  == 0, f"Can only mix an even number of samples. (batch size is {batch_size})"
        if batch_size % 2 != 0:
            h_x = h_x[:-1]
            y_pred = y_pred[:-1]
        mix_coeff = torch.rand(batch_size//2, dtype=x.dtype, device=x.device)

        h1 = h_x[0::2]
        h2 = h_x[1::2]
        mix_h_x = mixup(h1, h2, mix_coeff)
        
        y_pred_1 = y_pred[0::2]
        y_pred_2 = y_pred[1::2]
        y_pred_mix = mixup(y_pred_1, y_pred_2, mix_coeff)

        mix_y_pred = self.classifier(mix_h_x)

        loss = torch.dist(y_pred_mix, mix_y_pred)
        from .tasks import Tasks
        loss_info = LossInfo(
            name=Tasks.MANIFOLD_MIXUP,
            total_loss=loss,
            y_pred=y_pred_mix,
            y=mix_y_pred,
        )
        return loss_info
