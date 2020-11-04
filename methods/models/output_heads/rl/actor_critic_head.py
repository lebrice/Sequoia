""" TODO: *Actually* implement the actor-critic algorithm.

NOTE (@lebrice) This code here isn't meant as an actual implementation of the
actor-critic algo, it was more just me testing out stuff and writing what sortof
remember about actor-critic from my RL classes.
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Union
import gym
from gym import spaces
import torch
import numpy as np
from common.layers import Lambda
from torch import Tensor, nn, LongTensor
from utils.utils import prod
from settings.base.objects import Observations, Actions, Rewards
from .policy_head import PolicyHead, ClassificationOutput

# TODO: Refactor this to use a RegressionHead for the predicted reward and a
# ClassificationHead for the choice of action?


class ActorCriticHead(PolicyHead):
    def __init__(self,
                 input_size: int,
                 action_space: gym.Space,
                 reward_space: gym.Space,
                 hparams: "OutputHead.HParams" = None,
                 name: str = "classification"):
        assert isinstance(action_space, spaces.Discrete), "Only support discrete space for now."
        super().__init__(
            input_size=input_size,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        if not isinstance(input_size, int):
            input_size = prod(input_size)
        action_dims = prod(action_space.shape)

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
    def forward(self, observations: Observations, representations: Tensor) -> Actions:
        # NOTE: Here we could probably use either as the 'state':
        # state = observations.x 
        state = representations
        action = self.actor(state)
        # TODO: Actually implement the actor-critic forward pass. This is just a rough illustration.
        predicted_reward = self.critic([state, action])
        
        return {
            "action": action,
            "predicted_reward": predicted_reward,
        }


def concat_obs_and_action(observation_action: Tuple[Tensor, Tensor]) -> Tensor:
    observation, action = observation_action
    batch_size = observation.shape[0]
    observation = observation.reshape([batch_size, -1])
    action = action.reshape([batch_size, -1])
    return torch.cat([observation, action], dim=-1)
