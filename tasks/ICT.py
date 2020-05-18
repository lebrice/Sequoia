from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple
from utils.semi_sup.mean_teacher import architectures, datasets, data, losses, ramps, cli
import torch
import numpy as np
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import functional as F

from common.losses import LossInfo

from .auxiliary_task import AuxiliaryTask


def mixup_data_sup(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    # x, y = x.numpy(), y.numpy()
    # mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # y_a, y_b = torch.Tensor(y).type(torch.LongTensor), torch.Tensor(y[index]).type(torch.LongTensor)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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

class ICT(AuxiliaryTask):

    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Settings & Hyper-parameters related to the VAEReconstructionTask. """
        coefficient: float = 1.
        mixup_sup_alpha = 0.0

        consistency_type = 'mse'

    def __init__(self,
                 coefficient: float=None,
                 name: str="ict",
                 options: "ICT.Options"=None):
        super().__init__(coefficient=coefficient, name=name, options=options)
        self.class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
        if self.consistency_type == 'mse':
            self.consistency_criterion = losses.softmax_mse_loss
        elif self.consistency_type == 'kl':
            self.consistency_criterion = losses.softmax_kl_loss
        else:
            assert False, self.consistency_type

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> LossInfo:
        #x - all samples concat
        #h_x - all representations concat
        #y_pred - all predictions concat
        #y - labeles for sup x
        batch_size = x.shape[0]
        n_sup = len(y)

        x_sup= x[:n_sup]
        x_unsup = x[n_sup:]



        if x_sup.shape[0] != x_unsup.shape[0]:
            bt_size = np.minimum(input.shape[0], x_unsup.shape[0])
            x_sup = x_sup[0:bt_size]
            y = y[0:bt_size]
            x_unsup = x_unsup[0:bt_size]

        if self.mixup_sup_alpha:
            input_var, target_var, u_var = Variable(x_sup), Variable(y), Variable(x_unsup)
            mixed_input, target_a, target_b, lam = mixup_data_sup(x_sup, y, self.mixup_sup_alpha)
            mixed_input_var, target_a_var, target_b_var = Variable(mixed_input), Variable(target_a), Variable(target_b)
            output_mixed_l = self.classifier(self.encode(mixed_input_var))

            loss_func = mixup_criterion(target_a_var, target_b_var, lam)
            class_loss = loss_func(self.class_criterion, output_mixed_l)

        else:
            input_var = Variable(x_sup)
            with torch.no_grad():
                u_var = Variable(x_unsup)
            target_var = Variable(y(async = True))
            output = self.classifier(self.encode(input_var))
            class_loss = self.class_criterion(output, target_var)




        # assert batch_size % 2  == 0, f"Can only mix an even number of samples. (batch size is {batch_size})"
        if batch_size % 2 != 0:
            x = x[:-1]
            y_pred = y_pred[:-1]

        from .tasks import Tasks
        loss_info = LossInfo(name=Tasks.MIXUP)
        mix_coeff = torch.rand(batch_size//2, dtype=x.dtype, device=x.device)

        x1 = x[0::2]
        x2 = x[1::2]

        mix_x = mixup(x1, x2, mix_coeff)
        mix_h_x = self.encode(mix_x)
        mix_y_pred = self.classifier(mix_h_x)
        loss_info.tensors["mix_x"] = mix_x

        y_pred_1 = y_pred[0::2]
        y_pred_2 = y_pred[1::2]
        y_pred_mix = mixup(y_pred_1, y_pred_2, mix_coeff)
        loss_info.tensors["y_pred_mix"] = y_pred_mix

        loss = torch.dist(y_pred_mix, mix_y_pred)
        loss_info.total_loss = loss
        return loss_info