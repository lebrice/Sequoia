import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
from typing import Tuple
from sequoia.methods import Method
from simple_parsing import ArgumentParser
# TODO: Migrate stuff to directly import simple-parsing's hparams module.
# from simple_parsing.helpers.hparams import HyperParameters
from sequoia.common.hparams import HyperParameters
from sequoia.settings.active import (
    ActiveSetting,
    IncrementalRLSetting,
    ActiveEnvironment,
)
from torch import Tensor
from sequoia.settings import Observations, Actions, Rewards
from gym import spaces
from torch.distributions import Categorical
from gym.spaces.utils import flatten, flatdim, unflatten, flatten_space


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_size: int,
        learning_rate: float = 3e-4,
    ):
        super(ActorCritic, self).__init__()
        self.observation_space = observation_space
        # NOTE: See note below for why we don't use the task label portion of the space
        # here.
        self.num_inputs = flatdim(self.observation_space.x)
        self.hidden_size = hidden_size

        if not isinstance(action_space, spaces.Discrete):
            raise NotImplementedError(
                "This example only works with discrete action spaces."
            )
        self.action_space = action_space
        self.num_actions = self.action_space.n

        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_inputs, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1),
        )
        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_inputs, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.num_actions),
        )

    def forward(
        self, observation: ActiveSetting.Observations
    ) -> Tuple[Tensor, Categorical]:
        x = observation.x
        state = torch.as_tensor(x, dtype=torch.float)

        # NOTE: Here you could for instance concatenate the task labels onto the state
        # to make the model multi-task! However if you target the IncrementalRLSetting
        # or above, you might not have these task labels at test-time, so that would
        # have to be taken into consideration (e.g. can't concat None to a Tensor)
        # task_labels = observation.task_labels
        batched_inputs = state.ndim > 1
        if not batched_inputs:
            # Add a batch dimension if necessary.
            state = state.unsqueeze(0)

        value = self.critic(state)
        policy_logits = self.actor(state)

        if not batched_inputs:
            # Remove the batch dimension from the predictions if necessary.
            value = value.squeeze(0)
            policy_logits = policy_logits.squeeze(0)

        policy_dist = Categorical(logits=policy_logits)
        # policy_dist = F.relu(self.actor_linear1(state))
        # policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist


class ExampleA2CMethod(Method, target_setting=ActiveSetting):
    """ Example A2C method.

    Most of the code here was taken from https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
    """

    @dataclass
    class HParams(HyperParameters):
        # hyperparameters
        hidden_size: int = 256
        learning_rate: float = 1e-3

        # Constants
        GAMMA = 0.99
        num_steps = 300
        max_episodes = 3000
        entropy_term_coefficient = 0.001

    def __init__(self, hparams: HParams = None, render: bool = False):
        self.hparams = hparams or self.HParams()
        self.render: bool = True

    def configure(self, setting: ActiveSetting):
        self.num_inputs = setting.observation_space.x.shape[0]
        self.num_outputs = setting.action_space.n
        self.actor_critic = ActorCritic(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            hidden_size=self.hparams.hidden_size,
        )
        self.ac_optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.hparams.learning_rate
        )

    def fit(self, train_env: ActiveEnvironment, valid_env: ActiveEnvironment):
        assert isinstance(train_env, gym.Env)  # Just to illustrate that it's a gym Env.

        all_lengths = []
        average_lengths = []
        all_rewards = []

        for episode in range(self.hparams.max_episodes):
            log_probs = []
            values = []
            rewards = []
            entropy_term = 0

            observation: ActiveSetting.Observations = train_env.reset()
            for steps in range(self.hparams.num_steps):
                value, policy_dist = self.actor_critic.forward(observation)
                value = value.detach().numpy()
                action = policy_dist.sample()

                if self.render:
                    train_env.render()

                log_prob = policy_dist.log_prob(action)
                entropy = policy_dist.entropy()
                # NOTE: 'correct' thing to do would be to pass Actions objects of the
                # right type. This is for future-proofing this Method so it can
                # still function in the future if new settings are added.
                action = ActiveSetting.Actions(y_pred=action.detach().numpy())

                new_observation: ActiveSetting.Observations
                reward: ActiveSetting.Rewards
                new_observation, reward, done, _ = train_env.step(action)
                # Likewise, in order to support different future settings, we receive a
                # Rewards object, which contains the reward value (the float when the
                # env isn't batched.).
                reward_value: float = reward.y

                rewards.append(reward_value)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                observation = new_observation

                if done or steps == self.hparams.num_steps - 1:
                    Qval, _ = self.actor_critic.forward(new_observation)
                    Qval = Qval.detach().numpy()
                    all_rewards.append(np.sum(rewards))
                    all_lengths.append(steps)
                    average_lengths.append(np.mean(all_lengths[-10:]))
                    if episode % 10 == 0:
                        sys.stdout.write(
                            f"episode: {episode}, reward: {np.sum(rewards)}, "
                            f"total length: {steps}, "
                            f"average length: {average_lengths[-1]} \n"
                        )
                    break

            # compute Q values
            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.hparams.GAMMA * Qval
                Qvals[t] = Qval

            # update actor critic
            values = torch.as_tensor(values, dtype=torch.float)
            Qvals = torch.as_tensor(Qvals, dtype=torch.float)
            log_probs = torch.stack(log_probs)

            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + self.hparams.entropy_term_coefficient * entropy_term

            self.ac_optimizer.zero_grad()
            ac_loss.backward()
            self.ac_optimizer.step()

        # Plot results
        smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
        smoothed_rewards = [elem for elem in smoothed_rewards]
        plt.plot(all_rewards)
        plt.plot(smoothed_rewards)
        plt.plot()
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()

        plt.plot(all_lengths)
        plt.plot(average_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Episode length")
        plt.show()

    def get_actions(
        self, observations: ActiveSetting.Observations, action_space: gym.Space
    ) -> ActiveSetting.Actions:
        value, action_dist = self.actor_critic(observations)
        return ActiveSetting.Actions(y_pred=action_dist.sample())

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = ""):
        parser.add_arguments(cls.HParams, dest=(dest + "." if dest else "") + "hparams")

    @classmethod
    def from_argparse_args(cls, args, dest: str = ""):
        if dest:
            args = getattr(args, dest)
        hparams: ExampleA2CMethod.HParams = args.hparams
        return cls(hparams=hparams)


if __name__ == "__main__":
    from sequoia.settings.active import RLSetting

    setting = RLSetting(dataset="CartPole-v0", observe_state_directly=True, max_steps=1000, steps_per_task=1000)
    method = ExampleA2CMethod()

    results = setting.apply(method)
    print(results.summary())
