""" Global simclr like data augmentation  """
from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Tuple, Dict

import torch
import numpy as np
from torch import Tensor
from torch.autograd import Variable

from common.losses import LossInfo
from simple_parsing import mutable_field
from utils.logging_utils import get_logger  
from tasks.simclr.simclr_task import SimCLRTask

from ..addon import ExperimentAddon

logger = get_logger(__file__)

@dataclass
class Simclr_preprocess(ExperimentAddon):

    
    @dataclass 
    class Config(ExperimentAddon.Config):
        #whether to use SimCLR preprocessing globally
        simclr_augment_global: bool = False
    
    config: Config = mutable_field(Config)
    
    def preprocess_simclr_global(self, data:Tensor, target:Tensor=None, **kwargs) -> Tuple[Tensor, Tensor, Dict]:
        if self.config.simclr_augment_global:
            data, target, _ = SimCLRTask.preprocess_simclr(data,target, device=self.model.device)
        return data, target, kwargs


    