
"""Base class for an Agent interacting with an Active environments (ActiveDataLoader)

This is meant to be 'more general' than the 'Model' class, which is made for passive environments (regular dataloaders)
"""
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from pytorch_lightning import (EvalResult, LightningDataModule,
                               LightningModule, TrainResult)
from torch import Tensor

from common.config import Config
from settings import RLSetting
from settings.active import ActiveDataLoader
from utils.logging_utils import get_logger

from .hparams import HParams as BaseHParams
from .model import Model

logger = get_logger(__file__)
SettingType = TypeVar("SettingType", bound=RLSetting)

class Agent(LightningModule, Generic[SettingType]):
    """ LightningModule that interacts with `ActiveDataLoader` dataloaders.
    """
    @dataclass
    class HParams(BaseHParams):
        """ HParams of the Agent. """

    def __init__(self, setting: RLSetting, hparams: HParams, config: Config):
        super().__init__()
        # super().__init__(setting=setting, hparams=hparams, config=config)
        self.setting: RLSetting = setting
        self.datamodule: LightningDataModule = setting
        self.hp = hparams
        self.config = config
        
        # self.save_hyperparameters()

        self._train_loader: Optional[ActiveDataLoader] = None
        self._val_loader: Optional[ActiveDataLoader] = None
        self._test_loader: Optional[ActiveDataLoader] = None

        self.input_shape  = self.setting.obs_shape
        self.output_shape = self.setting.action_shape
        self.reward_shape = self.setting.reward_shape

        logger.debug(f"setting: {self.setting}")
        logger.debug(f"Input shape: {self.input_shape}")
        logger.debug(f"Output shape: {self.output_shape}")
        logger.debug(f"Reward shape: {self.reward_shape}")

        # Here we assume that all methods have a form of 'encoder' and 'output head'.
        # self.encoder, self.hidden_size = self.hp.make_encoder()
        # self.output_head = self.create_output_head()

        if self.config.debug and self.config.verbose:
            logger.debug("Config:")
            logger.debug(self.config.dumps(indent="\t"))
            logger.debug("Hparams:")
            logger.debug(self.hp.dumps(indent="\t"))
    
    def configure_optimizers(self):
        return self.hp.make_optimizer(self.parameters())

    def prepare_data(self, *args, **kwargs):
        self.setting.prepare_data(*args, **kwargs)

    def train_dataloader(self):
        self._train_loader = self.setting.train_dataloader()
        return self._train_loader
    
    def val_dataloader(self):
        self._val_loader = self.setting.val_dataloader()
        return self._val_loader

    def test_dataloader(self):
        self._test_loader = self.setting.test_dataloader()
        return self._test_loader

    def training_step(self, x: Tensor, *args,  **kwargs):
        logger.debug(f"Batch of observations of shape {x.shape}")
        actions = self.setting.train_env.random_actions()
        predictions = torch.as_tensor(actions, requires_grad=True)
        # predicted_reward = self.value(h_x, a_t)
        logger.debug(f"actions: {actions.shape}")
        rewards = self.setting.train_send(actions)
        logger.debug(f"Rewards: {rewards.shape}")
        loss = rewards.mean(dim=0)
        return TrainResult(loss)

    def validation_step(self, x: Tensor, *args, **kwargs):
        logger.debug(f"Batch of observations of shape {x.shape}")
        actions = self.setting.val_env.random_actions()
        logger.debug(f"actions: {actions.shape}")
        rewards = self.setting.val_send(actions)
        logger.debug(f"Rewards: {rewards.shape}")
        loss = rewards.mean(dim=0)
        return EvalResult(loss)

    def test_step(self, x: Tensor, *args, **kwargs):
        logger.debug(f"Batch of observations of shape {x.shape}")
        actions = self.setting.test_env.random_actions()
        logger.debug(f"actions: {actions.shape}")
        rewards = self.setting.test_send(actions)
        logger.debug(f"Rewards: {rewards.shape}")
        loss = rewards.mean(dim=0)
        return EvalResult(loss)
