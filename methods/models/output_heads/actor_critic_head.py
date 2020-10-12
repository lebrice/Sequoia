from typing import Dict, Tuple, Union

import torch
from common.layers import Lambda
from torch import Tensor, nn
from utils.utils import prod

from .output_head import OutputHead

# TODO: Refactor this to use a RegressionHead for the predicted reward and a
# ClassificationHead for the choice of action?


def concat_obs_and_action(observation_action: Tuple[Tensor, Tensor]) -> Tensor:
    observation, action = observation_action
    batch_size = observation.shape[0]
    observation = observation.reshape([batch_size, -1])
    action = action.reshape([batch_size, -1])
    return torch.cat([observation, action], dim=-1)


class ActorCriticHead(nn.Module):
    def __init__(self, input_size: Union[int, Tuple[int, ...]], action_dims: int):
        super().__init__()
        if not isinstance(input_size, int):
            input_size = prod(input_size)
        if not isinstance(action_dims, int):
            action_dims = prod(action_dims)

        self.critic_input_dims = input_size + action_dims
        self.critic_output_dims = 1
        self.critic = nn.Sequential(
            Lambda(concat_obs_and_action),
            nn.Linear(self.critic_input_dims, self.critic_output_dims),
        )
        self.actor_input_dims = input_size
        self.actor_output_dims = action_dims
        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.actor_input_dims, self.actor_output_dims),
        )

    # @auto_move_data
    def forward(self, state: Tensor) -> Dict[str, Tensor]:
        action = self.actor(state)
        predicted_reward = self.critic([state, action])
        return {
            "action": action,
            "predicted_reward": predicted_reward,
        }

