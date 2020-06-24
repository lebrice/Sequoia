""" Global data augmentation for mixup of supervised samples """
from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Tuple, Dict

import torch
import numpy as np
from torch import Tensor
from torch.autograd import Variable

from common.losses import LossInfo
from simple_parsing import mutable_field
from utils.logging_utils import get_logger

from ..addon import ExperimentAddon

logger = get_logger(__file__)

@dataclass
class MixUP_preprocess(ExperimentAddon):
    
    @dataclass 
    class Config(ExperimentAddon.Config):
        #for mixup of labeled data: the alpha parameter for the beta distribution from where the mixing lambda is drawn (mainly implemented for ICT)
        mixup_sup_alpha: float = 0.
    
    config: Config = mutable_field(Config)

    def preprocess_sup_mixup(self, x,y, mixup_sup_alpha):
        def mixup_criterion(y_a, y_b, lam):
            return lambda pred, _: lam * torch.nn.functional.cross_entropy(pred, y_a) + (1 - lam) * torch.nn.functional.cross_entropy(pred, y_b)
        mixed_input, target_a, target_b, lam = self.mixup_data_sup(x, y, mixup_sup_alpha)
        mixed_input_var, target_a_var, target_b_var = Variable(mixed_input), Variable(target_a), Variable(target_b)
        loss_func = mixup_criterion(target_a_var, target_b_var, lam)
        return mixed_input_var, loss_func

    @staticmethod
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
          
    def preprocess_mixup_supervised_global(self, data:Tensor, target:Tensor=None, **kwargs) -> Tuple[Tensor, Tensor, Dict]:
        if self.config.mixup_sup_alpha>0 and target is not None:
            data, supervised_criterion = self.preprocess_sup_mixup(data, target, self.config.mixup_sup_alpha)
            out = {'supervised_criterion':supervised_criterion}
            return data, target, dict(kwargs, **out)
        return data, target, kwargs


    