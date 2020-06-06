from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, Set, Dict, Optional

import torch
import numpy as np
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from common.losses import LossInfo
from common.task import Task

from .auxiliary_task import AuxiliaryTask
import logging
from config import Config
logger = logging.getLogger(__file__)

def mixup_data_sup(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    #x, y = x.numpy(), y.numpy()
    #mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_x = lam * x + (1 - lam) * x[index,:]
    #y_a, y_b = torch.Tensor(y).type(torch.LongTensor), torch.Tensor(y[index]).type(torch.LongTensor)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    x, y = x.data.cpu().numpy(), y.data.cpu().numpy()
    mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index, :])
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y[index, :])

    mixed_x = Variable(mixed_x.cuda())
    mixed_y = Variable(mixed_y.cuda())
    return mixed_x, mixed_y, lam

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction = 'sum') / num_classes


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


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
        mean_teacher_mixing_coefficient: float = 0.999
        # consistency_rampup_starts
        consistency_rampup_starts: int = 1
        # consistency_rampup_ends
        consistency_rampup_ends: int = 20
        # mixup_consistency
        mixup_consistency: float = 1.
        # for unsupervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn
        mixup_usup_alpha: float = 1.


    def get_current_consistency_weight(self, epoch, step_in_epoch, total_steps_in_epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        epoch = epoch - self.options.consistency_rampup_starts
        epoch = epoch + step_in_epoch / total_steps_in_epoch
        return self.options.mixup_consistency * sigmoid_rampup(epoch, self.options.consistency_rampup_ends - self.options.consistency_rampup_starts)

    def __init__(self,
                 coefficient: float=None,
                 name: str="Mixup",
                 options: "MixupTask.Options"=None):
        super().__init__(coefficient=coefficient, name=name, options=options)
        self.options: MixupTask.Options
        self.logger = Config.get_logger(__file__)

        # Exponential moving average versions of the encoder and output head.
        self.mean_encoder: nn.Module = deepcopy(AuxiliaryTask.encoder)
        self.mean_classifier: nn.Module = deepcopy(AuxiliaryTask.classifier)
        self.previous_task: Optional[Task] = None

        self.epoch_in_task: Optional[int] = 0
        self.epoch_length:  Optional[int] = 0
        self.update_number: Optional[int] = 0
        self.consistency_criterion = softmax_mse_loss

    def enable(self):
        self.mean_encoder = deepcopy(AuxiliaryTask.encoder)
        self.mean_classifier = deepcopy(AuxiliaryTask.classifier)

    def disable(self):
        del self.mean_encoder
        del self.mean_classifier

    def mean_encode(self, x: Tensor) -> Tensor:
        x, _ = AuxiliaryTask.preprocessing(x, None)
        return self.mean_encoder(x)

    def mean_logits(self, h_x: Tensor) -> Tensor:
        return self.mean_classifier(h_x)

    def on_model_changed(self, global_step: int, **kwargs)-> None:
        """ Executed when the model was updated. """
        self.epoch_in_task = kwargs.get('epoch')
        self.epoch_length = kwargs.get('epoch_length')
        self.update_number = kwargs.get('update_number')
        if self.enabled:
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

    def on_task_switch(self, task: Task, **kwargs) -> None:
        if self.enabled and task != self.previous_task:
            self.logger.info(f"Discarding the mean classifier on switch to task {task}")
            self.mean_classifier = deepcopy(AuxiliaryTask.classifier)
            self.previous_task = task

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> LossInfo:
        # select only unlabelled examples like in ICT: https://arxiv.org/pdf/1903.03825.pdf
        # TODO: fix this, y may be None, which would break this.

        batch_size = x.shape[0]
        # assert batch_size % 2  == 0, f"Can only mix an even number of samples. (batch size is {batch_size})"
        if batch_size % 2 != 0:
            x = x[:-1]
            y_pred = y_pred[:-1]

        from .tasks import Tasks
        loss_info = LossInfo(name=Tasks.MIXUP)

        if self.epoch_in_task < self.options.consistency_rampup_starts:
            mixup_consistency_weight = 0.0
        else:
            mixup_consistency_weight = self.get_current_consistency_weight(self.epoch_in_task,
                                                                           step_in_epoch=self.update_number,
                                                                           total_steps_in_epoch=self.epoch_length)
        if batch_size > 0 and mixup_consistency_weight > 0:
            #mix_coeff = torch.rand(batch_size//2, dtype=x.dtype, device=x.device)

            #x1 = x[0::2]
            #x2 = x[1::2]

            #mix_x = mixup(x1, x2, mix_coeff)

            #y_pred_1 = y_pred[0::2]
            #y_pred_2 = y_pred[1::2]


            h_x = self.mean_encode(x)
            y_pred_ema = self.mean_logits(h_x)
            mix_x, y_pred_mix, lam = mixup_data(x, Variable(y_pred_ema.detach().data, requires_grad=False), self.options.mixup_usup_alpha)

            loss_info.tensors["mix_x"] = mix_x.detach()
            mix_h_x = self.encode(mix_x)
            mix_y_pred = self.classifier(mix_h_x)

            # Use the mean teacher to get the h_x and y_pred for the unlabeled data.
            #h_x = self.mean_encode(x)
            #y_pred = self.mean_logits(h_x)
            #y_pred_mix = mixup(y_pred_1, y_pred_2, mix_coeff)


            loss_info.tensors["y_pred_mix"] = y_pred_mix.detach()
            loss = self.consistency_criterion(mix_y_pred, y_pred_mix) / batch_size  #
            #loss = torch.dist(y_pred_mix, mix_y_pred)
            loss_info.total_loss = mixup_consistency_weight * loss
        else:
            loss_info.total_loss = torch.tensor([0]).to(self.device)
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
