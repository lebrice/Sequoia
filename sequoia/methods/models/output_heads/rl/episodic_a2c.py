""" TODO: IDEA: Similar to ActorCriticHead, but episodic, i.e. only gives a Loss at
the end of the episode, rather than at each step.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Deque
import torch
import gym
import numpy as np
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


# TODO: Use this as inspiration: (taken from the ActorCriticPolicy from stable-baselines-3)
# NOTE: I see: so SB3 actually re-computes the values for everything in the
# rollout_data buffer every time! This is interesting.

# # (sb3 TODO): avoid second computation of everything because of the gradient
# values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
# values = values.flatten()

# # Normalize advantage (not present in the original implementation)
# advantages = rollout_data.advantages
# if self.normalize_advantage:
#     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# # Policy gradient loss
# policy_loss = -(advantages * log_prob).mean()

# # Value loss using the TD(gae_lambda) target
# value_loss = F.mse_loss(rollout_data.returns, values)

# # Entropy loss favor exploration
# if entropy is None:
#     # Approximate entropy when no analytical form
#     entropy_loss = -th.mean(-log_prob)
# else:
#     entropy_loss = -th.mean(entropy)

# loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

# # Optimization step
# self.policy.optimizer.zero_grad()
# loss.backward()

# # Clip grad norm
# th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

@dataclass(frozen=True)
class A2CHeadOutput(PolicyHeadOutput):
    """ Output produced by the A2C output head. """
    # The value estimate coming from the critic.
    value: Tensor

    @classmethod
    def stack(cls, items: List["A2CHeadOutput"]) -> "A2CHeadOutput":
        """TODO: Add a classmethod to 'stack' these objects. """


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
        entropy_loss_coef: float = 0.0

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
        self.critic = nn.Sequential(
            # Lambda(concat_obs_and_action),
            Flatten(),
            nn.Linear(self.critic_input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, self.critic_output_dims),
        )
        self.actions: List[Deque[A2CHeadOutput]]
        self._current_state: Optional[Tensor] = None
        self._previous_state: Optional[Tensor] = None
        self._step = 0
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @property
    def actor(self) -> nn.Module:
        return self.dense

    def forward(self,
                observations: ContinualRLSetting.Observations,
                representations: Tensor) -> A2CHeadOutput:
        return super().forward(observations, representations)

    def get_actions(self, representations: Tensor) -> A2CHeadOutput:
        # TODO: Shouldn't the critic also take the actor's action as an input?
        logits = self.dense(representations)
        action_distribution = Categorical(logits=logits)

        value = self.critic(representations)

        if action_distribution.has_rsample:
            sample = action_distribution.rsample()
        else:
            sample = action_distribution.sample()
        actions = A2CHeadOutput(
            y_pred=sample,
            logits=logits,
            action_dist=action_distribution,
            value=value,
        )
        return actions

    def create_buffers(self):
        super().create_buffers()
        
    def get_episode_loss(self, env_index: int, done: bool):
        inputs: Tensor
        actions: A2CHeadOutput
        rewards: Rewards
        # IDEA: Actually, now that I think about it, instead of detaching the
        # tensors, we could instead use the critic's 'value' estimate and get a
        # loss for that incomplete episode using the tensors in the buffer,
        # rather than detaching them!

        if not done:
            # For now, we only give back a loss at the end of the episode.
            # TODO: Test if giving back a loss at each step or every few steps
            # would work better!
            return None

        if len(self.actions[env_index]) == 0:
            raise RuntimeError(f"Weird, asked to get episode loss, but there is "
                               f"nothing in the buffer?")
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
        dones[-1] = done

        
        returns = self.get_returns(episode_rewards, gamma=self.hparams.gamma)
        advantages = returns - values

        # Normalize advantage (not present in the original implementation)
        if self.hparams.normalize_advantages:
            advantages = normalize(advantages)

        # Create the Loss to be returned.
        loss = Loss(self.name)
        
        # Policy gradient loss (actor loss)
        policy_gradient_loss = - (advantages * action_log_probs).mean()
        actor_loss = Loss("actor", policy_gradient_loss)
        loss += self.hparams.actor_loss_coef * actor_loss
        
        # Value loss: Try to get the critic's values close to the actual return.
        value_loss_tensor = F.mse_loss(returns, values)
        critic_loss = Loss("critic", value_loss_tensor)
        loss += self.hparams.actor_loss_coef * actor_loss

        entropy_loss = Loss("entropy", actions.action_dist.entropy().mean())
        loss += self.hparams.entropy_loss_coef * entropy_loss
        
        loss.metric = EpisodeMetrics(rewards=episode_rewards.tolist())
        loss.metrics["gradient_usage"] = self.get_gradient_usage_metrics(env_index)

        return loss
        # # Value loss using the TD(gae_lambda) target
        # value_loss = F.mse_loss(rollout_data.returns, values)

        # # Entropy loss favor exploration
        # if entropy is None:
        #     # Approximate entropy when no analytical form
        #     entropy_loss = -th.mean(-log_prob)
        # else:
        #     entropy_loss = -th.mean(entropy)

        # loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        # # Optimization step
        # self.policy.optimizer.zero_grad()
        # loss.backward()

        # # Clip grad norm
        # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)



        # Construct the 'q' values from the right to the left.
        # The estimated value for the final state in the episode.
        # q_val = values[-1]
        # q_vals: Deque[Tensor] = deque(maxlen=None)
        # for reward, done_i in list(zip(episode_rewards, dones))[::-1]:
        #     q_val = reward + (~done_i) * self.hparams.gamma * q_val
        #     q_vals.appendleft(q_val) # store values from the end to the beginning
        # advantage = torch.stack(list(q_vals)) - values
        # assert False, advantage
        # loss = Loss(self.name)
        
        # critic_loss_tensor = advantage.pow(2).mean()
        # critic_loss = Loss("critic", critic_loss_tensor)
        # loss += critic_loss
                
        # actor_loss_tensor = (- action_log_probs * advantage.detach()).mean()
        # actor_loss = Loss("actor", actor_loss_tensor)
        # loss += actor_loss
        
        # loss.metric = RLMetrics(episodes=[EpisodeMetrics(rewards=episode_rewards.tolist())])
        
        # # Calculate how many of the inputs had gradients.
        # gradient_usage = self.get_gradient_usage_metrics(env_index)
        # loss.metrics["gradient_usage"] = gradient_usage
        # return loss
    
    
    def stack_buffers(self, env_index: int):
        """ Stack the observations/actions/rewards for this env and return them.
        """
        # episode_observations = tuple(self.observations[env_index])
        episode_representations: List[Tensor] = list(self.representations[env_index])
        episode_actions: List[A2CHeadOutput] = list(self.actions[env_index])
        episode_rewards: List[Rewards] = list(self.rewards[env_index])
        # TODO: Could maybe use out=<some parameter on this module> to
        # prevent having to create new 'container' tensors at each step?

        # Make sure this all still works (should work even better) once we
        # change the obs spaces to dicts instead of Tuples.
        assert len(episode_representations)
        assert len(episode_actions)
        assert len(episode_rewards)
        stacked_inputs = stack(self.input_space, episode_representations)
        # stacked_actions = stack(self.action_space, episode_actions)
        # stacked_rewards = stack(self.reward_space, episode_rewards)
        episode_length = len(stacked_inputs)
        # TODO: Update this to use 'stack' if we change the action/reward spaces
        y_preds = torch.stack([action.y_pred for action in episode_actions])
        logits = torch.stack([action.logits for action in episode_actions])
        values = torch.stack([action.value for action in episode_actions])
        values = values.reshape([episode_length])

        stacked_actions = A2CHeadOutput(
            y_pred=y_preds,
            logits=logits,
            action_dist=Categorical(logits=logits),
            value=values,
        )
        rewards_type = type(episode_rewards[0])
        assert all(reward.y is not None for reward in episode_rewards)
        stacked_rewards = rewards_type(
            y=torch.stack([reward.y for reward in episode_rewards])  # type: ignore
        )
        return stacked_inputs, stacked_actions, stacked_rewards


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
