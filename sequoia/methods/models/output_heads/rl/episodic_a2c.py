""" TODO: IDEA: Similar to ActorCriticHead, but episodic, i.e. only gives a Loss at
the end of the episode, rather than at each step.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Deque
import torch
import gym
import numpy as np
from simple_parsing import mutable_field
from gym import Space, spaces
from gym.spaces.utils import flatdim
from sequoia.common.layers import Flatten
from sequoia.common import Loss
from sequoia.settings import ContinualRLSetting
from sequoia.settings.base import Rewards
from torch import Tensor, nn
from torch.nn import functional as F
from sequoia.utils.generic_functions import detach, get_slice, set_slice, stack

from .policy_head import Categorical, PolicyHead, PolicyHeadOutput, GradientUsageMetric
from .policy_head import normalize
from sequoia.common.metrics.rl_metrics import EpisodeMetrics, RLMetrics
from sequoia.utils import get_logger

logger = get_logger(__file__)



@dataclass(frozen=True)
class A2CHeadOutput(PolicyHeadOutput):
    """ Output produced by the A2C output head. """
    # The value estimate coming from the critic.
    value: Tensor

class EpisodicA2C(PolicyHead):
    """ Advantage-Actor-Critic output head that produces a loss only at end of
    episode.
    
    TODO: This could actually produce a loss every N steps, rather than just at
    the end of the episode.
    """

    @dataclass
    class HParams(PolicyHead.HParams):
        """ Hyper-parameters of the episodic A2C output head. """
        # Wether to normalize the advantages for each episode.
        normalize_advantages: bool = False

        actor_loss_coef: float = 0.5
        critic_loss_coef: float = 0.5
        entropy_loss_coef: float = 0.1

        # Maximum norm of the policy gradient.
        max_policy_grad_norm: Optional[float] = None

        # The discount factor.
        gamma: float = 0.99

        # Learning rate of the optimizer.
        learning_rate: float = 1e-2

    def __init__(self,
                 input_space: spaces.Box,
                 action_space: spaces.Discrete,
                 reward_space: spaces.Box,
                 hparams: HParams = None,
                 name: str = "episodic_a2c"):
        super().__init__(
            input_space=input_space,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        self.hparams: EpisodicA2C.HParams
        # Critic takes in state-action pairs? or just state?
        self.critic_input_dims = self.input_size
        # self.critic_input_dims = self.input_size + action_dims
        self.critic_output_dims = 1
        self.critic = self.make_dense_network(
            in_features=self.critic_input_dims,
            hidden_neurons=self.hparams.hidden_neurons,
            out_features=self.critic_output_dims,
            activation=self.hparams.activation,
        )
        self.actions: List[Deque[A2CHeadOutput]]
        self._current_state: Optional[Tensor] = None
        self._previous_state: Optional[Tensor] = None
        self._step = 0

    @property
    def actor(self) -> nn.Module:
        return self.dense

    def forward(self,
                observations: ContinualRLSetting.Observations,
                representations: Tensor) -> A2CHeadOutput:
        actions: PolicyHeadOutput = super().forward(observations, representations)
        # TODO: Shouldn't the critic also take the actor's action as an input?
        value = self.critic(representations)
        # We just need to add the value to the actions of the PolicyHead.
        # This works, because `self.actor` :== `self.dense`, which is what's used by
        # the PolicyHead.
        actions = A2CHeadOutput(
            y_pred=actions.y_pred,
            logits=actions.logits,
            action_dist=actions.action_dist,
            value=value
        )
        return actions

    def num_stored_steps(self, env_index: int) -> int:
        """ Returns the number of steps stored in the buffer for the given
        environment index.
        """
        return len(self.actions[env_index])

    def get_episode_loss(self, env_index: int, done: bool):
        inputs: Tensor
        actions: A2CHeadOutput
        rewards: Rewards
        # IDEA: Actually, now that I think about it, instead of detaching the
        # tensors, we could instead use the critic's 'value' estimate and get a
        # loss for that incomplete episode using the tensors in the buffer,
        # rather than detaching them!

        if not done:
            return None

        # TODO: Add something like a 'num_steps_since_update' for each env? (it
        # would actually be a num_steps_since_backward)
        # if self.num_steps_since_update?

        if self.num_stored_steps(env_index) < 5:
            # For now, we only give back a loss at the end of the episode.
            # TODO: Test if giving back a loss at each step or every few steps
            # would work better!
            return None

        inputs, actions, rewards = self.stack_buffers(env_index)
        logits: Tensor = actions.logits
        action_log_probs: Tensor = actions.action_log_prob
        values: Tensor = actions.value
        assert rewards.y is not None
        episode_rewards: Tensor = rewards.y
        
        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        episode_length = len(episode_rewards)
        dones = torch.zeros(episode_length, dtype=torch.bool)
        dones[-1] = bool(done)

        returns = self.get_returns(episode_rewards, gamma=self.hparams.gamma).type_as(values)
        advantages = returns - values

        # Normalize advantage (not present in the original implementation)
        if self.hparams.normalize_advantages:
            advantages = normalize(advantages)

        # Create the Loss to be returned.
        loss = Loss(self.name)

        # Policy gradient loss (actor loss)
        policy_gradient_loss = - (advantages.detach() * action_log_probs).mean()
        actor_loss = Loss("actor", policy_gradient_loss)
        loss += self.hparams.actor_loss_coef * actor_loss
        
        # Value loss: Try to get the critic's values close to the actual return,
        # which means the advantages should be close to zero.
        value_loss_tensor = F.mse_loss(values, returns.reshape(values.shape))
        critic_loss = Loss("critic", value_loss_tensor)
        loss += self.hparams.critic_loss_coef * critic_loss

        # Entropy loss, to "favor exploration".
        entropy_loss = Loss("entropy", - actions.action_dist.entropy().mean())
        loss += self.hparams.entropy_loss_coef * entropy_loss
        
        if done:
            episode_metrics = [EpisodeMetrics(rewards=episode_rewards.tolist())]
            loss.metric = RLMetrics(episodes=episode_metrics)
        
        loss.metrics["gradient_usage"] = self.get_gradient_usage_metrics(env_index)

        return loss

    def optimizer_step(self):
        # Clip grad norm if desired.
        if self.hparams.max_policy_grad_norm is not None:
            original_norm: Tensor = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.hparams.max_policy_grad_norm,
            )
            self.loss.metrics["policy_gradient_norm"] = original_norm.item()
        super().optimizer_step()


def compute_returns_and_advantage(self, last_values: Tensor, dones: np.ndarray) -> None:
    """
    TODO: Adapting this snippet from SB3's common/buffers.py RolloutBuffer.

    Post-processing step: compute the returns (sum of discounted rewards)
    and GAE advantage.
    Adapted from Stable-Baselines PPO2.

    Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
    to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
    where R is the discounted reward with value bootstrap,
    set ``gae_lambda=1.0`` during initialization.

    :param last_values:
    :param dones:

    """
    buffer_size: int = self.buffer_size
    dones: np.ndarray = self.dones
    rewards: np.ndarray = self.rewards
    values: np.ndarray = self.values
    gamma: float = self.gamma
    gae_lambda: float = 1.0
    # convert to numpy
    last_values = last_values.clone().cpu().numpy().flatten()
    advantages = np.zeros_like(rewards)

    last_gae_lam = 0
    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - dones
            next_values = last_values
        else:
            next_non_terminal = 1.0 - dones[step + 1]
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        self.advantages[step] = last_gae_lam
    self.returns = self.advantages + self.values
