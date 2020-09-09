# Random agent.
from typing import Dict, Tuple, Union, List

import numpy as np
import torch
from pytorch_lightning.core.decorators import auto_move_data
from torch import Tensor, nn

from common.config import Config
from common.loss import Loss
from settings import ActiveSetting, RLSetting, Setting
from utils import prod, try_get

from .actor_critic_agent import ActorCritic
from .agent import Agent
from .output_heads import OutputHead


class RandomAgent(ActorCritic):
    """Version of ActorCritic where the action taken and the predicted values
    are random.
    """

    @auto_move_data
    def forward(self, batch: Union[Tensor, Tuple[Tensor, Tensor, List[bool], Dict]]) -> Dict[str, Tensor]:
        if isinstance(batch, tuple):
            assert False, "TODO: don't know how to handle batch"
        else:
            state = batch
        x = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        # No real need to encode the value here, since the action and predicted
        # rewards are random, but we do it just in case this is used to debug a
        # method like Self-Supervision which needs to an `h_x`.
        h_x = self.encode(x)

        action = self.get_action(x)
        predicted_reward = self.get_value(x, action)
        return dict(
            x=x,
            h_x=h_x,
            action=action,
            predicted_reward=predicted_reward,
        )

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
