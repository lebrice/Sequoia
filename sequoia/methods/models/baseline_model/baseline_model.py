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
from gym import Space, spaces
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.core.lightning import ModelSummary, log
from simple_parsing import Serializable, choice, mutable_field
from torch import Tensor, nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from sequoia.settings import Setting, ActiveSetting, PassiveSetting
from sequoia.common.config import Config
from sequoia.common.loss import Loss
from sequoia.methods.aux_tasks.auxiliary_task import AuxiliaryTask
from sequoia.methods.models.output_heads import (ActorCriticHead,
                                                 ClassificationHead,
                                                 OutputHead, PolicyHead,
                                                 RegressionHead)
from sequoia.settings import ContinualRLSetting
from sequoia.utils.logging_utils import get_logger

torch.autograd.set_detect_anomaly(True)

logger = get_logger(__file__)


# WIP (@lebrice): Playing around with this idea, to try and maybe use the idea
# of creating typed objects for the 'Observation', the 'Action' and the 'Reward'
# for each kind of model.
from sequoia.settings import Actions, Observations, Rewards
from sequoia.settings.assumptions.incremental import IncrementalSetting
SettingType = TypeVar("SettingType", bound=IncrementalSetting)

from .base_model import ForwardPass
from .class_incremental_model import ClassIncrementalModel
from .self_supervised_model import SelfSupervisedModel
from .semi_supervised_model import SemiSupervisedModel


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
    class HParams(SemiSupervisedModel.HParams,
                  SelfSupervisedModel.HParams,
                  ClassIncrementalModel.HParams):
        """ HParams of the Model. """
        # NOTE: The different hyper-parameters can be found as fields in the
        # base classes, but most of them are defined in base_hparams.py.
        pass

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

        if self.trainer:
            OutputHead.base_model_optimizer = self.optimizers()

    # @auto_move_data
    def forward(self, observations: IncrementalSetting.Observations) -> ForwardPass:
        # NOTE: Implementation is mostly in `base_model.py`.
        # Share the optimizer with the output head, in case the representations
        # are also learned with them.
        OutputHead.base_model_optimizer = self.optimizers()
        return super().forward(observations)

    def create_output_head(self, setting: Setting, add_to_optimizer: bool = None) -> OutputHead:
        """Create an output head for the current setting.
        
        NOTE: This assumes that the input, action and reward spaces don't change
        between tasks.
        
        Parameters
        ----------
        add_to_optimizer : bool, optional
            Wether to add the parameters of the new output head to the optimizer
            of the model. Defaults to None, in which case we add the output head
            parameters as long as it doesn't have an `optimizer` attribute of
            its own.

        Returns
        -------
        OutputHead
            The new output head.
        """
        # NOTE: Implementation is in `base_model.py`.
        return super().create_output_head(setting, add_to_optimizer=add_to_optimizer)

    def output_head_type(self, setting: SettingType) -> Type[OutputHead]:
        """ Return the type of output head we should use in a given setting.
        """
        # NOTE: Implementation is in `base_model.py`.
        return super().output_head_type(setting)

    def training_step(self,
                      batch: Tuple[Observations, Optional[Rewards]],
                      batch_idx: int,
                      *args, **kwargs):
        step_result = self.shared_step(
            batch,
            batch_idx,
            environment=self.setting.train_env,
            loss_name="train",
            *args,
            **kwargs
        )
        loss: Tensor = step_result["loss"]
        loss_object: Loss = step_result["loss_object"]
        # return step_result
        
        if loss != 0.:    
            self.log("train loss", loss, on_step=True, prog_bar=True, logger=True)

            for key, value in loss_object.to_pbar_message().items():
                self.log(key, value, on_step=True, prog_bar=True)
                logger.debug(f"{key}: {value}")

            for key, value in loss_object.to_log_dict().items():
                self.log(key, value, on_step=True, logger=True)

        if not isinstance(loss, Tensor) or not loss.requires_grad:
            # TODO: There might be no loss at some steps, because for instance
            # we haven't reached the end of an episode in an RL setting.
            return None

        if loss.requires_grad:
            
            if isinstance(self.output_head, PolicyHead):
                # In this case we don't do the step, since the output head has
                # its own optimizer. TODO: Should probably improve the
                # online_a2c such that we can get a loss at each step always,
                # which would make this much simpler.
                optimizer = self.optimizers()
                self.manual_backward(loss, optimizer, retain_graph=True)
  
            elif not self.trainer.train_loop.automatic_optimization:
                optimizer = self.optimizers()
                self.manual_backward(loss, optimizer)
                optimizer.step()
                optimizer.zero_grad()
        return step_result
        
    def validation_step(self,
                        batch: Tuple[Observations, Optional[Rewards]],
                        batch_idx: int,
                        *args,
                        **kwargs):
        return self.shared_step(
            batch,
            batch_idx,
            environment=self.setting.val_env,
            loss_name="val",
            *args,
            **kwargs,
        )

    def test_step(self,
                  batch: Tuple[Observations, Optional[Rewards]],
                  batch_idx: int,
                  *args,
                  **kwargs):
        return self.shared_step(
            batch,
            batch_idx,
            *args,
            environment=self.setting.test_env,
            loss_name="test",
            **kwargs,
        )
