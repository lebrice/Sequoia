# Random agent.


import torch
from torch import Tensor

from common.config import Config
from common.loss import Loss
from settings import ActiveSetting, RLSetting, Setting

from .actor_critic_agent import ActorCritic
from .agent import Agent


class RandomAgent(ActorCritic):
    """Version of ActorCritic where the action taken and the predicted values
    are random.
    """
    def get_value(self, observation: Tensor, action: Tensor) -> Tensor:
        return torch.rand(
            self.setting.reward_shape,
            requires_grad=True,
            dtype=self.dtype,
            device=self.device,
        )

    def get_action(self, observation: Tensor) -> Tensor:
        actions = self.setting.val_env.random_actions()
        actions = torch.as_tensor(actions, dtype=self.dtype, device=self.device)
        return actions
