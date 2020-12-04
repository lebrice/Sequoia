""" Example/Template of a Model to be used as part of a Method.

You can use this as a base class when creating your own models, or you can
start from scratch, whatever you like best.
"""
from dataclasses import dataclass
from typing import *

import gym
import numpy as np
import pytorch_lightning as pl
import torch
from gym import spaces
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.core.lightning import ModelSummary, log
from torch import Tensor, nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from common.config import Config
from common.loss import Loss
from methods.aux_tasks.auxiliary_task import AuxiliaryTask
from methods.models.output_heads import OutputHead, ClassificationHead, RegressionHead, PolicyHead
from simple_parsing import Serializable, choice, mutable_field
from utils.logging_utils import get_logger

logger = get_logger(__file__)
SettingType = TypeVar("SettingType", bound=LightningDataModule)


# WIP (@lebrice): Playing around with this idea, to try and maybe use the idea
# of creating typed objects for the 'Observation', the 'Action' and the 'Reward'
# for each kind of model.
from settings import Observations, Actions, Rewards
from settings.assumptions.incremental import IncrementalSetting
from .base_model import ForwardPass
from .semi_supervised_model import SemiSupervisedModel
from .class_incremental_model import ClassIncrementalModel
from .self_supervised_model import SelfSupervisedModel

class BaselineModel(SemiSupervisedModel,
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
        SemiSupervisedModel.HParams,
        SelfSupervisedModel.HParams,
        ClassIncrementalModel.HParams,
    ):
        """ HParams of the Model. """

    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)

        self.save_hyperparameters({
            "hparams": self.hp.to_dict(),
            "config": self.config.to_dict(),
        })
        logger.debug(f"setting of type {type(self.setting)}")
        logger.debug(f"Observation space: {self.observation_space}")
        logger.debug(f"Action/Output space: {self.action_space}")
        logger.debug(f"Reward/Label space: {self.reward_space}")

        if self.config.debug and self.config.verbose:
            logger.debug("Config:")
            logger.debug(self.config.dumps(indent="\t"))
            logger.debug("Hparams:")
            logger.debug(self.hp.dumps(indent="\t"))


    # @auto_move_data
    def forward(self, observations: IncrementalSetting.Observations) -> Dict[str, Tensor]:
        return super().forward(observations)

    def training_step(self,
                      batch: Tuple[Observations, Optional[Rewards]],
                      batch_idx: int,
                      *args,
                      **kwargs):
        step_result = self.shared_step(
            batch,
            batch_idx,
            *args,
            environment=self.setting.train_env,
            loss_name="train",
            **kwargs
        )
        # FIXME: Debugging the baseline method on RL settings. 
        from settings.active.rl import ContinualRLSetting
        assert isinstance(batch, ContinualRLSetting.Observations)
        observations = batch
        loss: Tensor = step_result["loss"]
        if any(observations.done):
            assert False, (batch_idx, observations, loss)
        # assert False, (batch, loss)
        if loss == 0.:
            return None
        
        # TODO: Figure this out.
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss, optimizer)
        assert False, loss
        self.manual_backward(loss, optimizer)
        # self.log("")
        self.manual_optimizer_step(optimizer)
        return loss


    def validation_step(self,
                      batch: Tuple[Observations, Optional[Rewards]],
                      *args,
                      **kwargs):
        return self.shared_step(
            batch,
            *args,
            environment=self.setting.val_env,
            loss_name="val",
            **kwargs
        )

    def test_step(self,
                      batch: Tuple[Observations, Optional[Rewards]],
                      *args,
                      **kwargs):
        return self.shared_step(
            batch,
            *args,
            environment=self.setting.test_env,
            loss_name="test",
            **kwargs
        )
    