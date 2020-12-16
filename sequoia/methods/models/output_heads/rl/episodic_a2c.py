""" TODO: IDEA: Similar to ActorCriticHead, but episodic, i.e. only gives a Loss at
the end of the episode, rather than at each step.
"""
from collections import deque
from dataclasses import dataclass
from typing import List, Optional
import torch
import gym
from gym import Space, spaces
from gym.spaces.utils import flatdim
from sequoia.common.layers import Flatten
from sequoia.common import Loss
from sequoia.settings import ContinualRLSetting
from sequoia.settings.base import Rewards
from torch import Tensor, nn
from sequoia.utils.generic_functions import detach, get_slice, set_slice, stack

from .policy_head import Categorical, PolicyHead, PolicyHeadOutput, GradientUsageMetric
from sequoia.common.metrics.rl_metrics import EpisodeMetrics, RLMetrics

@dataclass(frozen=True)
class A2CHeadOutput(PolicyHeadOutput):
    # The value estimate coming from the critic.
    value: Tensor 


class EpisodicA2C(PolicyHead):
    def __init__(self,
                 input_space: spaces.Box,
                 action_space: spaces.Discrete,
                 reward_space: spaces.Box,
                 hparams: "PolicyHead.HParams" = None,
                 name: str = "episodic_a2c"):
        super().__init__(
            input_space=input_space,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
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
        
        self.actions: List[A2CHeadOutput]
        self._current_state: Optional[Tensor] = None
        self._previous_state: Optional[Tensor] = None
        self._step = 0
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

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
            policy=action_distribution,
            value=value,
        )
        return actions
    
    def create_buffers(self):
        super().create_buffers()
        # self.values = [
        #     deque(maxlen=self.hparams.max_episode_window_length) for i in range(self.batch_size)
        # ]
        
    def get_episode_loss(self, env_index: int, done: bool):
        inputs: Tensor
        actions: A2CHeadOutput
        rewards: Rewards
        # IDEA: Actually, now that I think about it, instead of detaching the
        # tensors, we could instead use the critic's 'value' estimate and get a
        # loss for that incomplete episode using the tensors in the buffer,
        # rather than detaching them!

        if not done:
            # This particular algorithm (REINFORCE) can't give a loss until the
            # end of the episode is reached.
            return None

        if len(self.actions[env_index]) == 0:
            raise RuntimeError(f"Weird, asked to get episode loss, but there is "
                               f"nothing in the buffer?")

        inputs, actions, rewards = self.stack_buffers(env_index)
        logits: Tensor = actions.logits
        action_log_probs: Tensor = actions.action_log_prob
        values: Tensor = actions.value
        episode_rewards: Tensor = rewards.y
        
        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        episode_length = len(episode_rewards)
        dones = torch.zeros(episode_length, dtype=torch.bool)
        dones[-1] = done
        
        # last_q_val: The estimated value for the final state in the episode.
        q_val = values[-1]
        q_vals: List[Tensor] = deque(maxlen=None)
        for reward, done_i in list(zip(episode_rewards, dones))[::-1]:
            q_val = reward + (~done_i) * self.hparams.gamma * q_val
            q_vals.appendleft(q_val) # store values from the end to the beginning

        advantage = torch.stack(list(q_vals)) - values
        
        loss = Loss(self.name)
        
        critic_loss_tensor = advantage.pow(2).mean()
        # adam_critic.zero_grad()
        # critic_loss.backward()
        # adam_critic.step()
        critic_loss = Loss("critic", critic_loss_tensor)
        loss += critic_loss
                
        actor_loss_tensor = (- action_log_probs * advantage.detach()).mean()
        # adam_actor.zero_grad()
        # actor_loss.backward()
        # adam_actor.step()
        actor_loss = Loss("actor", actor_loss_tensor)
        loss += actor_loss
        
        loss.metric = RLMetrics(episodes=[EpisodeMetrics(rewards=episode_rewards.tolist())])
        
        # Calculate how many of the inputs had gradients.
        episode_actions = self.actions[env_index]
        n_stored_items = len(self.actions[env_index])
        n_items_with_grad = sum(v.logits.requires_grad for v in episode_actions)
        n_items_without_grad = n_stored_items - n_items_with_grad
        self.num_grad_tensors[env_index] += n_items_with_grad
        self.num_detached_tensors[env_index] += n_items_without_grad
        # logger.debug(f"Env {env_index} produces a loss based on "
        #              f"{n_items_with_grad} tensors with grad and "
        #              f"{n_items_without_grad} without. ")

        gradient_usage = GradientUsageMetric(
            used_gradients=n_items_with_grad,
            wasted_gradients=n_items_without_grad,
        )
        loss.metrics["gradient_usage"] = gradient_usage
        return loss
    
    
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

        # TODO: Update this to use 'stack' if we change the action/reward spaces
        y_preds = torch.stack([action.y_pred for action in episode_actions])
        logits = torch.stack([action.logits for action in episode_actions])
        values = torch.stack([action.value for action in episode_actions])
        stacked_actions = A2CHeadOutput(
            y_pred=y_preds,
            logits=logits,
            policy=Categorical(logits=logits),
            value=values,
        )
        rewards_type = type(episode_rewards[0])
        stacked_rewards = rewards_type(
            y=stack(self.reward_space, [reward.y for reward in episode_rewards])
        )
        return stacked_inputs, stacked_actions, stacked_rewards

