""" An output head for RL based on Advantage Actor Critic.

NOTE: This is the 'online' version of an Advantage Actor Critic, based
on the following blog:

https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b

"""

from dataclasses import dataclass
from typing import Dict, Tuple, Union, Optional

import gym
import numpy as np
import torch
from gym import spaces
from gym.spaces.utils import flatdim
from torch import LongTensor, Tensor, nn
from torch.optim.optimizer import Optimizer

from sequoia.common.layers import Lambda, Flatten
from sequoia.common import Loss
from sequoia.settings.base.objects import Actions, Observations, Rewards
from sequoia.settings import ContinualRLSetting
from sequoia.utils.utils import prod
from sequoia.utils import get_logger

from ...forward_pass import ForwardPass
from ..classification_head import ClassificationOutput, ClassificationHead
from .policy_head import PolicyHead, PolicyHeadOutput, Categorical
logger = get_logger(__file__)

class ActorCriticHead(ClassificationHead):
    
    @dataclass
    class HParams(ClassificationHead.HParams):
        """ Hyper-parameters of the Actor-Critic head. """
        gamma: float = 0.95
        learning_rate: float = 1e-3

    def __init__(self,
                 input_space: spaces.Space,
                 action_space: spaces.Discrete,
                 reward_space: spaces.Box,
                 hparams: "ActorCriticHead.HParams" = None,
                 name: str = "actor_critic"):
        assert isinstance(action_space, spaces.Discrete), "Only support discrete space for now."
        super().__init__(
            input_space=input_space,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        if not isinstance(self.hparams, self.HParams):
            self.hparams = self.upgrade_hparams()
            
        action_dims = flatdim(action_space)

        # Critic takes in state-action pairs? or just state?
        self.critic_input_dims = self.input_size
        # self.critic_input_dims = self.input_size + action_dims
        self.critic_output_dims = 1
        self.critic = nn.Sequential(
            # Lambda(concat_obs_and_action),
            Flatten(),
            nn.Linear(self.critic_input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, self.critic_output_dims),
        )
        self.actor_input_dims = self.input_size
        self.actor_output_dims = action_dims
        self.actor = nn.Sequential(
            Flatten(),
            nn.Linear(self.actor_input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, self.actor_output_dims),
        )        
        self._current_state: Optional[Tensor] = None
        self._previous_state: Optional[Tensor] = None
        self._step = 0

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hparams.learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.learning_rate)

    def forward(self,
                observations: ContinualRLSetting.Observations,
                representations: Tensor) -> PolicyHeadOutput:
        # NOTE: Here we could probably use either as the 'state':
        # state = observations.x
        # state = representations
        representations = representations.float()
        if len(representations.shape) != 2:
            representations = representations.reshape([-1, self.actor_input_dims])
        
        self._previous_state = self._current_state
        self._current_state = representations
        
        # TODO: Actually implement the actor-critic forward pass.
        # predicted_reward = self.critic([state, action])
        # Do we want to detach the representations? or not?
        
        logits = self.actor(representations)
        # The policy is the distribution over actions given the current state.
        action_dist = Categorical(logits=logits)
        
        if action_dist.has_rsample:
            sample = action_dist.rsample()
        else:
            sample = action_dist.sample()

        actions = PolicyHeadOutput(
            y_pred=sample,
            logits=logits,
            action_dist=action_dist,
        )
        return actions
  
    def get_loss(self,
                 forward_pass: ForwardPass,
                 actions: PolicyHeadOutput,
                 rewards: ContinualRLSetting.Rewards) -> Loss:
        action_dist: Categorical = actions.action_dist

        rewards = rewards.to(device=actions.device)
        env_reward = torch.as_tensor(rewards.y, device=actions.device)

        observations: ContinualRLSetting.Observations = forward_pass.observations
        done = observations.done
        assert done is not None, "Need the end-of-episode signal!"
        done = torch.as_tensor(done, device=actions.device)
        assert self._current_state is not None
        if self._previous_state is None:
            # Only allow this once!
            assert self._step == 0
            self._previous_state = self._current_state
        self._step += 1

        # TODO: Need to detach something here, right?
        advantage: Tensor = (
            env_reward
            +  (~done) * self.hparams.gamma * self.critic(self._current_state)
            - self.critic(self._previous_state) # detach previous representations?
        )
        
        total_loss = Loss(self.name)
        if self.training:
            self.optimizer_critic.zero_grad()
        critic_loss_tensor = (advantage ** 2).mean()
        critic_loss = Loss("critic", loss=critic_loss_tensor)
        if self.training:
            critic_loss_tensor.backward()
            self.optimizer_critic.step()
            
        total_loss += critic_loss.detach()

        if self.training:
            self.optimizer.zero_grad()
        actor_loss_tensor = - action_dist.log_prob(actions.action) * advantage.detach()
        actor_loss_tensor = actor_loss_tensor.mean()
        actor_loss = Loss("actor", loss=actor_loss_tensor)
        if self.training:
            actor_loss_tensor.backward()
            self.optimizer.step()

        total_loss += actor_loss.detach()

        return total_loss


def concat_obs_and_action(observation_action: Tuple[Tensor, Tensor]) -> Tensor:
    observation, action = observation_action
    batch_size = observation.shape[0]
    observation = observation.reshape([batch_size, -1])
    action = action.reshape([batch_size, -1])
    return torch.cat([observation, action], dim=-1)
