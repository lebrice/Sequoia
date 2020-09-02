
"""Base class for an Agent interacting with an Active environments (ActiveDataLoader)

This is meant to be 'more general' than the 'Model' class, which is made for passive environments (regular dataloaders)
"""
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Tuple, TypeVar

import torch
from pytorch_lightning import (EvalResult, LightningDataModule,
                               LightningModule, TrainResult)
from torch import Tensor, nn

from common.config import Config
from common.layers import Lambda
from common.loss import Loss
from settings import RLSetting
from settings.active import ActiveDataLoader
from settings.active.rl import GymDataLoader
from utils import prod
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

        self.loss_fn: Callable[[Tensor, Tensor], Tensor] = torch.dist

        critic_input_dims = prod(self.input_shape) + prod(self.setting.action_shape)
        critic_output_dims = prod(self.reward_shape)

        # assert False, critic_input_dims
        self.critic = nn.Sequential(
            Lambda(concat_obs_and_action),
            nn.Linear(critic_input_dims, critic_output_dims),
        )
        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(self.input_shape), prod(self.output_shape)),
        )

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
        return self.setting.train_dataloader()
    
    def val_dataloader(self):
        return self.setting.val_dataloader()

    def test_dataloader(self):
        return self.setting.test_dataloader()

    def get_value(self, observation: Tensor, action: Tensor) -> Tensor:
        # assert False, (observation.shape, observation.dtype)
        observation = torch.as_tensor(observation, dtype=self.dtype, device=self.device)
        action = torch.as_tensor(action, dtype=self.dtype, device=self.device)
        assert observation.shape[0] == action.shape[0], (observation.shape, action.shape)
        return self.critic([observation, action])

        return torch.rand(self.reward_shape, requires_grad=True)
    
    def get_action(self, observation: Tensor) -> Tensor:
        # assert False, (observation.shape, observation.dtype)
        actions = self.setting.val_env.random_actions()
        actions = torch.as_tensor(actions, dtype=self.dtype, device=self.device)
        return actions

        observation = torch.as_tensor(observation, dtype=self.dtype, device=self.device)
        return self.actor(observation)

    def shared_step(self, observation: Tensor, environment: GymDataLoader, loss_name: str) -> Dict:
        logger.debug(f"Batch of observations of shape {observation.shape}")

        # Get the action to perform
        actions = self.get_action(observation)
        logger.debug(f"actions: {actions.shape}")

        # Get the predicted reward for that action.
        predicted_rewards = self.get_value(observation, actions)
        logger.debug(f"predicted rewards shape: {predicted_rewards.shape}")

        # Get the actual reward for that action.
        rewards = environment.send(actions)
        logger.debug(f"Rewards shape: {rewards.shape}, dtype: {rewards.dtype}, device: {rewards.device}")
        rewards = rewards.to(self.device)
        
        # TODO: Calculate an actual loss that makes sense. Just playing around
        # for now.
        nce = self.loss_fn(predicted_rewards, rewards)
        loss = Loss(loss_name, loss=nce) #y_pred=predicted_rewards, y=rewards)
        # return loss.to_pl_dict()
        # Trying to figure out what to return:

        result = TrainResult(minimize=loss.loss)
        result.log("mean_reward", loss.loss)
        # result["loss_object"] = loss
        return result

    def training_step(self,
                      batch: Tensor,
                      batch_idx: int,
                      optimizer_idx: int = None,
                      *args,
                      **kwargs):
        """
        Args:
            observation (Tensor): The observation from the environment.
            batch_idx (int): Integer displaying index of this batch
            optimizer_idx (int, optional): When using multiple optimizers, this
                argument will also be present.
        """
        return self.shared_step(
            batch,
            environment=self.setting.train_env,
            loss_name="train",
        )

    def validation_step(self,
                        batch: Tensor,
                        batch_idx: int,
                        optimizer_idx: int = None,
                        *args,
                        **kwargs):
        return self.shared_step(
            batch,
            environment=self.setting.val_env,
            loss_name="val",
        )

    def test_step(self,
                  batch: Tensor,
                  batch_idx: int,
                  optimizer_idx: int = None,
                  *args,
                  **kwargs):
        return self.shared_step(
            batch,
            environment=self.setting.test_env,
            loss_name="test",
        )

def concat_obs_and_action(observation_action: Tuple[Tensor, Tensor]) -> Tensor:
    observation, action = observation_action
    batch_size = observation.shape[0]
    observation = observation.reshape([batch_size, -1])
    action = action.reshape([batch_size, -1])
    return torch.cat([observation, action], dim=-1)
