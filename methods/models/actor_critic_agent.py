from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor, nn

from common.config import Config
from common.layers import Lambda
from common.loss import Loss
from settings import RLSetting
from utils import prod
from utils.logging_utils import get_logger

from .agent import Agent

logger = get_logger(__file__)


class ActorCritic(Agent):
    @dataclass
    class HParams(Agent.HParams):
        """ HyperParameters of the Actor-Critic Agent. """
        # TODO: Add the hyper-parameters from actor-critic here.

    def __init__(self, setting: RLSetting, hparams: "ActorCritic.HParams", config: Config):
        super().__init__(setting, hparams, config)

        # Actor-critic related stuff:

        critic_input_dims = prod(self.input_shape) + prod(self.setting.action_shape)
        critic_output_dims = prod(self.reward_shape)

        self.loss_fn: Callable[[Tensor, Tensor], Tensor] = torch.dist
        self.critic = nn.Sequential(
            Lambda(concat_obs_and_action),
            nn.Linear(critic_input_dims, critic_output_dims),
        )
        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(self.input_shape), prod(self.output_shape)),
        )

    def forward(self, batch: Union[Tensor, Tuple[Tensor, Tensor, List[bool], Dict]]) -> Tensor:
        # TODO: What about methods that want to compare the current 'state' and
        # the next 'state'? How would we pass the 'previous state' to it?
        logger.debug(f"batch len: {len(batch)}")
        if isinstance(batch, tuple):
            assert False, "TODO: don't know how to handle batch"
        else:
            observations = batch

        logger.debug(f"Batch of observations of shape {observations.shape}")
        action = self.get_action(observations)
        predicted_reward = self.get_value(observations, action)
        return {
            "action": action,
            "predicted_reward": predicted_reward,
        }

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

    def get_loss(self, forward_pass: Dict[str, Tensor], reward: Tensor = None, loss_name: str = "") -> Loss:
        # TODO: What about methods that want to compare the current 'state' and
        # the next 'state'? How would we pass the 'previous state' to it?
        action = forward_pass["action"]
        predicted_reward = forward_pass["predicted_reward"]
        total_loss: Loss = Loss(loss_name)  
               
        nce = self.loss_fn(predicted_reward, reward)
        critic_loss = Loss("critic", loss=nce)
        total_loss += critic_loss
        # TODO: doing this from memory, actor-critic is definitely not that simple.
        actor_loss = Loss("actor", loss=-predicted_reward.mean())
        total_loss += actor_loss

        return total_loss


def concat_obs_and_action(observation_action: Tuple[Tensor, Tensor]) -> Tensor:
    observation, action = observation_action
    batch_size = observation.shape[0]
    observation = observation.reshape([batch_size, -1])
    action = action.reshape([batch_size, -1])
    return torch.cat([observation, action], dim=-1)
