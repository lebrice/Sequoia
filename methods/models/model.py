""" Example/Template of a Model to be used as part of a Method.

You can use this as a base class when creating your own models, or you can
start from scratch, whatever you like best.
"""
from dataclasses import dataclass
from typing import *

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.core.lightning import ModelSummary, log
from torch import Tensor, nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from common.config import Config
from common.loss import Loss
from common.tasks.auxiliary_task import AuxiliaryTask
from methods.models.output_heads import OutputHead
from simple_parsing import Serializable, choice, mutable_field
from utils.logging_utils import get_logger

logger = get_logger(__file__)
SettingType = TypeVar("SettingType", bound=LightningDataModule)


# WIP (@lebrice): Playing around with this idea, to try and maybe use the idea
# of creating typed objects for the 'Observation', the 'Action' and the 'Reward'
# for each kind of model.
from .base_model import ForwardPass
from settings import Observations, Actions, Rewards
from .model_addons import SemiSupervisedModel, ClassIncrementalModel, SelfSupervisedModel


class Model(SemiSupervisedModel,
            ClassIncrementalModel,
            SelfSupervisedModel,
            Generic[SettingType]):
    """ Base model LightningModule (nn.Module extended by pytorch-lightning)
    
    This model splits the learning task into a representation-learning problem
    and a downstream task (output head) applied on top of it.   

    The most important method to understand is the `get_loss` method, which
    is used by the [train/val/test]_step methods which are called by
    pytorch-lightning.
    """

    @dataclass
    class HParams(
        # SemiSupervisedModel.HParams,
        SelfSupervisedModel.HParams,
        ClassIncrementalModel.HParams,
    ):
        """ HParams of the Model. """

    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)

        logger.debug(f"setting of type {type(self.setting)}")
        logger.debug(f"Observation space: {self.observation_space}")
        logger.debug(f"Action/Output space: {self.action_space}")
        logger.debug(f"Reward/Label space: {self.reward_space}")
        
        if self.config.debug and self.config.verbose:
            logger.debug("Config:")
            logger.debug(self.config.dumps(indent="\t"))
            logger.debug("Hparams:")
            logger.debug(self.hp.dumps(indent="\t"))
