""" LR linear rampu and cosine rampdown """
from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Tuple

import torch
import numpy as np
from torch import Tensor
from torch.autograd import Variable

from common.losses import LossInfo
from simple_parsing import mutable_field
from utils.logging_utils import get_logger 
from tasks.simclr.simclr_task import SimCLRTask

from .addon import ExperimentAddon

logger = get_logger(__file__)

@dataclass
class LRannealer(ExperimentAddon):
    @dataclass  
    class Config(ExperimentAddon.Config):
        #lr scheduling: length of learning rate rampup in the beginning (mainly for ICT)
        lr_rampup: int = 5
        #length of learning rate cosine rampdown (>= length of training): the epoch at which learning rate reaches to zero (mainly for ICT)
        lr_rampdown_epochs: int = 0 
    config: Config = mutable_field(Config)

    @staticmethod
    def cosine_rampdown(current, rampdown_length):
        """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
        assert 0 <= current <= rampdown_length
        return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
    @staticmethod
    def linear_rampup(current, rampup_length):
        """Linear rampup"""
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            lr = 1.0
        else:
            lr = current / rampup_length

        # print (lr)
        return lr
        
    def adjust_learning_rate(self, optimizer, epoch, step_in_epoch, total_steps_in_epoch, epochs):
        current_lr = self.hparams.learning_rate   #self.current_lr
        initial_lr = 0.0 #self.hparams.learning_rate
        epoch = epoch + step_in_epoch / total_steps_in_epoch

        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = self.linear_rampup(epoch, self.config.lr_rampup) * (current_lr - initial_lr) + initial_lr

        # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
        if self.config.lr_rampdown_epochs:
            assert self.config.lr_rampdown_epochs >= epochs
            lr *= self.cosine_rampdown(epoch, self.config.lr_rampdown_epochs)
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr