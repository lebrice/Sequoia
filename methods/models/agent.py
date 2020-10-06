
"""Base class for an Agent interacting with an Active environments (ActiveDataLoader)

This is meant to be 'more general' than the 'Model' class, which is made for passive environments (regular dataloaders)
"""
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (Callable, Dict, Generic, List, Optional, Tuple, TypeVar,
                    Union)
from gym.spaces import Discrete, Box
import numpy as np
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
from utils import prod, try_get
from utils.serialization import Pickleable
from utils.logging_utils import get_logger

from .base_hparams import BaseHParams
from .model import Model
from .output_heads import OutputHead

logger = get_logger(__file__)
SettingType = TypeVar("SettingType", bound=RLSetting)

class Agent(Model[SettingType]):
    """ LightningModule that interacts with `ActiveDataLoader` dataloaders.
    """
    @dataclass
    class HParams(Model.HParams):
        """ HParams of the Agent. """

    def __init__(self, setting: RLSetting, hparams: HParams, config: Config):
        # super().__init__()
        super().__init__(setting=setting, hparams=hparams, config=config)
        self.setting: RLSetting
        assert self.setting.dims == self.setting.obs_shape, (self.setting.dims, self.setting.obs_shape)
        self.input_shape = self.setting.obs_shape
        if isinstance(self.setting.action_space, Discrete):
            self.action_shape = (self.setting.action_space.n,)
        elif isinstance(self.setting.action_space, Box):
            self.action_shape = self.setting.action_space.shape

        self.output_shape = self.action_shape
        self.reward_shape = self.setting.reward_shape
        self.total_reward: Tensor = 0.  # type: ignore

        self.output_head: OutputHead = self.create_output_head()
    
    def create_output_head(self) -> OutputHead:
        """ Create the output head for the task. """
        return OutputHead(self.hidden_size, self.output_shape, name="policy")

    def configure_optimizers(self):
        return self.hp.make_optimizer(self.parameters())

    @abstractmethod
    def forward(self, batch) -> Dict[str, Tensor]:
        """ Forward pass of your model.
        
        Preprocess inputs and create all the tensors required for the backward
        pass here.

        In the case of RL, this must return a dict with the next action to take,
        at one of the 'action', 'actions' or 'y_pred' keys.
        
        This could look like this, for example:
        ```
        x = self.preprocess_x(observation)
        h_x = self.encode(x)
        y_pred = self.output_head(h_x)
        return {
            "x": x,
            "h_x": h_x,
            "y_pred": y_pred,
        }
        ```
        """

    @abstractmethod
    def get_loss(self, forward_pass: Dict[str, Tensor], reward: Tensor = None, loss_name: str = "") -> Loss:
        """Gets a Loss given the results of the forward pass and the reward.00

        Args:
            forward_pass (Dict[str, Tensor]): Results of the forward pass.
            reward (Tensor, optional): The reward that resulted from the action
                chosen in the forward pass. Defaults to None.
            loss_name (str, optional): The name for the resulting Loss.
                Defaults to "".

        Returns:
            Loss: a Loss object containing the loss tensor, associated metrics
            and sublosses.
        
        This could look a bit like this, for example:
        ```
        action = forward_pass["action"]
        predicted_reward = forward_pass["predicted_reward"]
        nce = self.loss_fn(predicted_reward, reward)
        loss = Loss(loss_name, loss=nce)
        return loss
        ```
        """

    def select_action(self, forward_pass: Dict[str, Tensor]) -> np.ndarray:
        action = try_get(forward_pass, "action", "actions", "y_pred")
        if action is None:
            raise RuntimeError("The dict returned by `forward()` must include "
                               "either a 'action' or 'y_pred' entry.")
        action_array = action.detach().cpu().numpy()
        if isinstance(self.setting.action_space, Discrete):
            return action_array.argmax(-1)
        return action_array

    def shared_step(self,
                    batch: Tuple[Tensor, Tensor],
                    batch_idx: int,
                    environment: GymDataLoader,
                    loss_name: str,
                    dataloader_idx: int = None,
                    optimizer_idx: int = None,
                    ) -> Dict:

        # Process the observation, encode it, create whatever tensors you want.
        forward_pass = self.forward(batch)
        # Extract the action to take from the forward pass dict.
        actions = self.select_action(forward_pass)
               
        # Send the action to the environment, get back the associated reward.
        logger.debug(f"Sending actions to the environment: {actions}")
        # TODO: Need to format the action (make it back into an int or
        # something) so we select what to do.
        rewards = environment.send(actions)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=self.dtype)
        # rewards = rewards.to(self.device, dtype=self.dtype)

        # Get a loss to backpropagate. This should ideally be a Loss object.
        loss: Loss = self.get_loss(forward_pass, rewards, loss_name=loss_name)
        assert isinstance(loss, Loss), f"get_loss should return a Loss object for now. (received {loss})"
        
        if batch_idx == 0:
            self.total_reward = 0.
        self.total_reward += rewards.mean().detach()
        mean_reward = self.total_reward / (batch_idx or 1)

        result_dict = loss.to_pl_dict()
        result_dict["log"]["mean_reward"] = mean_reward
        return result_dict
        # TODO: I don't really understand how the TrainResult and EvalResult
        # objects are supposed to be used.
        # if self.train:
        #     result = TrainResult(loss)
        # else:
        #     result = EvalResult()
        # result.log("n_steps", torch.as_tensor(float(batch_idx)), prog_bar=True)
        # result.log("mean_reward", mean_reward, prog_bar=True)
        # result.log("rewards", rewards.mean().detach(), prog_bar=True)
        # # result.log("predicted rewards", predicted_rewards.mean().detach(), prog_bar=True)
        # # result["loss_object"] = loss
        # return result
        # return loss.to_pl_dict()

    def training_step(self,
                      batch: Tensor,
                      batch_idx: int,
                      *args,
                      optimizer_idx: int = None,
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
                        *args,
                        optimizer_idx: int = None,
                        dataloader_idx: int = None,
                        **kwargs):
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
                  *args,
                  optimizer_idx: int = None,
                  dataloader_idx: int = None,
                  **kwargs):
        return self.shared_step(
            batch,
            batch_idx=batch_idx,
            environment=self.setting.test_env,
            optimizer_idx=optimizer_idx,
            dataloader_idx=dataloader_idx,
            loss_name="test",
        )

    @contextmanager
    def using_environment(self, environment: GymDataLoader) -> None:
        # Idea: we could maybe use such a context manager to avoid having to
        # switch between environments (or having to pass the environment to the
        # `shared_step`).
        # TODO: Not used atm.
        starting_env: GymDataLoader = self.current_env
        self.current_env = environment
        yield self.current_env
        self.current_env = starting_env
