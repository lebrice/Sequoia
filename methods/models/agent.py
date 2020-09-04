
"""Base class for an Agent interacting with an Active environments (ActiveDataLoader)

This is meant to be 'more general' than the 'Model' class, which is made for passive environments (regular dataloaders)
"""
from contextlib import contextmanager
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

        self.total_reward: Tensor = 0.  # type: ignore
        
        # Here we assume that all methods have a form of 'encoder' and 'output head'.
        # self.encoder, self.hidden_size = self.hp.make_encoder()
        # self.output_head = self.create_output_head()
        if self.config.debug and self.config.verbose:
            logger.debug("Config:")
            logger.debug(self.config.dumps(indent="\t"))
            logger.debug("Hparams:")
            logger.debug(self.hp.dumps(indent="\t"))


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

    def configure_optimizers(self):
        return self.hp.make_optimizer(self.parameters())

    def get_value(self, observation: Tensor, action: Tensor) -> Tensor:
        # FIXME: This is here just for debugging purposes.  
        # assert False, (observation.shape, observation.dtype)
        observation = torch.as_tensor(observation, dtype=self.dtype, device=self.device)
        action = torch.as_tensor(action, dtype=self.dtype, device=self.device)
        assert observation.shape[0] == action.shape[0], (observation.shape, action.shape)
        return self.critic([observation, action])

        return torch.rand(self.reward_shape, requires_grad=True)
    
    def get_action(self, observation: Tensor) -> Tensor:
        # FIXME: This is here just for debugging purposes.
        # assert False, (observation.shape, observation.dtype)
        actions = self.setting.val_env.random_actions()
        actions = torch.as_tensor(actions, dtype=self.dtype, device=self.device)
        return actions

        observation = torch.as_tensor(observation, dtype=self.dtype, device=self.device)
        return self.actor(observation)

    def shared_step(self,
                    batch: Tuple[Tensor, Tensor],
                    batch_idx: int,
                    environment: GymDataLoader,
                    loss_name: str,
                    dataloader_idx: int = None,
                    optimizer_idx: int = None,
                    ) -> Dict:
        logger.debug(f"batch len: {len(batch)}")
        if len(batch) == 2:
            # TODO: How should we handle this 'previous reward' ?
            observations, previous_rewards = batch
        else:
            observations = batch

        logger.debug(f"Batch of observations of shape {observations.shape}")
        # Get the action to perform
        actions = self.get_action(observations)
        logger.debug(f"actions: {actions.shape}")
        
        # Get the actual reward for that action.
        rewards = environment.send(actions)

        if self.config.debug:
            import matplotlib.pyplot as plt
            plt.ion()
            environment.environments[0].render(mode="human")
            # plt.draw()
        # breakpoint()

        logger.debug(f"Rewards shape: {rewards.shape}, dtype: {rewards.dtype}, device: {rewards.device}")
        rewards = rewards.to(self.device)

        # Get the predicted reward for that action.
        predicted_rewards = self.get_value(observations, actions)
        logger.debug(f"predicted rewards shape: {predicted_rewards.shape}")

        # TODO: Calculate an actual loss that makes sense. Just debugging atm.
        nce = self.loss_fn(predicted_rewards, rewards)
        
        loss = Loss(loss_name, loss=nce) #y_pred=predicted_rewards, y=rewards)
        # breakpoint()
        # Trying to figure out what to return:
        if loss.name == "train":
            result = TrainResult(minimize=loss.loss)
        else:
            result = EvalResult()

        self.total_reward += rewards.mean().detach()
        
        mean_reward = self.total_reward / (batch_idx or 1)
    
        # result.log("loss", loss.loss, prog_bar=True)
        result.log("n_steps", torch.as_tensor(float(batch_idx)), prog_bar=True)
        result.log("mean_reward", mean_reward, prog_bar=True)
        result.log("rewards", rewards.mean().detach(), prog_bar=True)
        result.log("predicted rewards", predicted_rewards.mean().detach(), prog_bar=True)
        # result["loss_object"] = loss
        return result
        # return loss.to_pl_dict()

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
            batch_idx=batch_idx,
            environment=self.setting.train_env,
            optimizer_idx=optimizer_idx,
            loss_name="train",
        )

    def validation_step(self,
                        batch: Tensor,
                        batch_idx: int,
                        optimizer_idx: int = None,
                        dataloader_idx: int = None,
                        *args,
                        **kwargs):
        super().validation_step()
        return self.shared_step(
            batch,
            batch_idx=batch_idx,
            environment=self.setting.val_env,
            optimizer_idx=optimizer_idx,
            dataloader_idx=dataloader_idx,
            loss_name="val",
        )

    def test_step(self,
                  batch: Tensor,
                  batch_idx: int,
                  optimizer_idx: int = None,
                  dataloader_idx: int = None,
                  *args,
                  **kwargs):
        return self.shared_step(
            batch,
            environment=self.setting.test_env,
            batch_idx=batch_idx,
            optimizer_idx=optimizer_idx,
            dataloader_idx=dataloader_idx,
            loss_name="test",
        )

    @contextmanager
    def using_environment(self, environment: GymDataLoader) -> None:
        # Idea: we could maybe use such a context manager to avoid having to
        # switch between environments (or having to pass the environment to the
        # `shared_step`).
        starting_env: GymDataLoader = self.current_env
        self.current_env = environment
        yield self.current_env
        self.current_env = starting_env

    
    def forward(self, observation: Tensor) -> Dict[str, Tensor]:
        # TODO: something like this:
        x = self.preprocess_x(observation)
        h_x = self.encode(x)
        y_pred = self.output_head(h_x)
        return {
            "x": x,
            "h_x": h_x,
            "y_pred": y_pred,
        }

    def get_loss(self, observation: Tensor, reward: Tensor = None, **forward_pass: Dict[str, Tensor]) -> Loss:
        # TODO: Figure out a way to merge / refactor the 'get_loss' function,
        # in such a way that works for RL and also for supervised learning.
        x = forward_pass["x"]
        h_x = forward_pass["h_x"]
        y_pred = forward_pass["y_pred"]
        # (...)
        loss = Loss(loss_name, loss=nce)
        return loss
    
        

def concat_obs_and_action(observation_action: Tuple[Tensor, Tensor]) -> Tensor:
    observation, action = observation_action
    batch_size = observation.shape[0]
    observation = observation.reshape([batch_size, -1])
    action = action.reshape([batch_size, -1])
    return torch.cat([observation, action], dim=-1)
