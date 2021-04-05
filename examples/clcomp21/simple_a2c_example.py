import sys
from dataclasses import dataclass
from pathlib import Path
import itertools
from typing import Dict, List, Optional, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from gym.spaces.utils import flatdim
from sequoia.common.spaces import Image
from sequoia.common.hparams import HyperParameters, log_uniform, uniform
from sequoia.methods import Method
from sequoia.settings.active import ActiveEnvironment, ActiveSetting

# TODO: Migrate stuff to directly import simple-parsing's hparams module.
# from simple_parsing.helpers.hparams import HyperParameters
from simple_parsing import ArgumentParser
from torch import Tensor
from torch.distributions import Categorical
from torch.utils.data import DataLoader


class ActorCritic(nn.Module):
    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, hidden_size: int,
    ):
        super().__init__()
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

        if self.num_inputs < 100:
            # If we have a reasonably-small input space, use an MLP architecture.
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
        else:
            assert isinstance(self.observation_space.x, Image)
            channels = self.observation_space.x.channels
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 6, kernel_size=5, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(6),
                nn.ReLU(inplace=True),
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.AdaptiveAvgPool2d(output_size=(8, 8)),  # [16, 8, 8]
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(32),  # [32, 6, 6]
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(32),  # [32, 4, 4]
                nn.Flatten(),
            )
            # NOTE: Here we share the encoder for both the actor and critic.
            self.critic = nn.Sequential(
                self.encoder,
                nn.Linear(512, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 1),
            )
            self.actor = nn.Sequential(
                self.encoder,
                nn.Linear(512, self.hidden_size),
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
        x_space = self.observation_space.x
        batched_inputs = state.ndim > len(x_space.shape)
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

    Most of the code here was taken from:
    https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
    """

    @dataclass
    class HParams(HyperParameters):
        """ Hyper-Parameters of the model, as a dataclass.

        Fields get command-line arguments with simple-parsing.
        """

        # Hidden size (representation size).
        hidden_size: int = 256
        # Learning rate of the optimizer.
        learning_rate: float = log_uniform(1e-6, 1e-2, default=3e-4)
        # Discount factor
        gamma: float = 0.99
        # Coefficient for the entropy term in the loss formula.
        entropy_term_coefficient: float = 0.001
        # Maximum length of an episode, when desired. (Generally not needed).
        max_episode_steps: Optional[int] = None

    def __init__(self, hparams: HParams = None, render: bool = False):
        self.hparams = hparams or self.HParams()
        self.task: int = 0
        self.plots_dir: Path = Path("plots")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.render = render

    def configure(self, setting: ActiveSetting):
        self.actor_critic = ActorCritic(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            hidden_size=self.hparams.hidden_size,
        ).to(self.device)
        self.ac_optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.hparams.learning_rate
        )
        # If there is a limit on the number of steps per task, then observe that limit.
        self.max_training_steps = setting.steps_per_phase

    def fit(self, train_env: ActiveEnvironment, valid_env: ActiveEnvironment):
        assert isinstance(train_env, gym.Env)  # Just to illustrate that it's a gym Env.

        # NOTE: This example only works if the environment isn't vectorized.

        all_lengths: List[int] = []
        average_lengths: List[float] = []
        all_rewards: List[float] = []
        episode = 0
        total_steps = 0

        while (
            not train_env.is_closed()
            and total_steps < self.max_training_steps
        ):
            episode += 1

            log_probs: List[Tensor] = []
            values: List[Tensor] = []
            rewards: List[Tensor] = []
            entropy_term = 0

            observation: ActiveSetting.Observations = train_env.reset()
            # Convert numpy arrays in the observation into Tensors on the right device.
            observation = observation.torch(device=self.device)

            done = False
            episode_steps = 0
            while not done and total_steps < self.max_training_steps:
                episode_steps += 1
                value, policy_dist = self.actor_critic.forward(observation)
                value = value.cpu().detach().numpy()
                action = policy_dist.sample()

                log_prob = policy_dist.log_prob(action)
                entropy = policy_dist.entropy()
                # NOTE: 'correct' thing to do would be to pass Actions objects of the
                # right type. This is for future-proofing this Method so it can
                # still function in the future if new settings are added.
                action = ActiveSetting.Actions(y_pred=action.cpu().detach().numpy())

                if self.render:
                    train_env.render()

                new_observation: ActiveSetting.Observations
                reward: ActiveSetting.Rewards
                new_observation, reward, done, _ = train_env.step(action)
                new_observation = new_observation.torch(device=self.device)
                total_steps += 1

                # Likewise, in order to support different future settings, we receive a
                # Rewards object, which contains the reward value (the float when the
                # env isn't batched.).
                reward_value: float = reward.y

                rewards.append(reward_value)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy

                observation = new_observation

            Qval, _ = self.actor_critic.forward(new_observation)
            Qval = Qval.detach().cpu().numpy()
            all_rewards.append(np.sum(rewards))
            all_lengths.append(episode_steps)
            average_lengths.append(np.mean(all_lengths[-10:]))

            if episode % 10 == 0:
                print(
                    f"step {total_steps}/{self.max_training_steps}, "
                    f"episode: {episode}, "
                    f"reward: {np.sum(rewards)}, "
                    f"total length: {episode_steps}, "
                    f"average length: {average_lengths[-1]} \n"
                )

            if total_steps >= self.max_training_steps:
                print(
                    f"Reached the limit of {self.max_training_steps} steps."
                )
                break

            # compute Q values
            Q_values = np.zeros_like(values)
            # Use the last value from the critic as the final value estimate.
            q_value = Qval
            for t, reward in reversed(list(enumerate(rewards))):
                q_value = reward + self.hparams.gamma * q_value
                Q_values[t] = q_value

            # update actor critic
            values = torch.as_tensor(values, dtype=torch.float, device=self.device)
            Q_values = torch.as_tensor(Q_values, dtype=torch.float, device=self.device)
            log_probs = torch.stack(log_probs)

            advantage = Q_values - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = (
                actor_loss
                + critic_loss
                + self.hparams.entropy_term_coefficient * entropy_term
            )

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
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plots_dir / f"task_{self.task}_0.png")
        # plt.show()

        plt.plot(all_lengths)
        plt.plot(average_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Episode length")
        plt.savefig(self.plots_dir / f"task_{self.task}_1.png")
        # plt.show()

    def get_actions(
        self, observations: ActiveSetting.Observations, action_space: gym.Space
    ) -> ActiveSetting.Actions:
        # Move the observations to the right device, converting numpy arrays to tensors.
        observations = observations.torch(device=self.device)
        value, action_dist = self.actor_critic(observations)
        return ActiveSetting.Actions(y_pred=action_dist.sample())

    # The methods below aren't required, but are good to add.

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Called by the Setting when switching between tasks.

        Parameters
        ----------
        task_id : Optional[int]
            the id of the new task. When None, we are
            basically being informed that there is a task boundary, but without
            knowing what task we're switching to.
        """
        if isinstance(task_id, int):
            self.task = task_id

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = ""):
        parser.add_arguments(cls.HParams, dest=(dest + "." if dest else "") + "hparams")

    @classmethod
    def from_argparse_args(cls, args, dest: str = ""):
        if dest:
            args = getattr(args, dest)
        hparams: ExampleA2CMethod.HParams = args.hparams
        return cls(hparams=hparams)

    def get_search_space(self, setting: ActiveSetting) -> Dict:
        return self.hparams.get_orion_space()

    def adapt_to_new_hparams(self, new_hparams: Dict) -> None:
        self.hparams = self.HParams.from_dict(new_hparams)


if __name__ == "__main__":

    # Create the Setting.

    # CartPole-state for debugging:
    from sequoia.settings.active import RLSetting

    setting = RLSetting(dataset="CartPole-v0", observe_state_directly=True)

    # OR: Incremental CartPole-state:
    from sequoia.settings.active import IncrementalRLSetting

    setting = IncrementalRLSetting(
        dataset="CartPole-v0",
        observe_state_directly=True,
        monitor_training_performance=True,
        nb_tasks=1,
        steps_per_task=1_000,
        test_steps=2000,
    )

    # OR: Setting of the RL Track of the competition:
    # setting = IncrementalRLSetting.load_benchmark("rl_track")

    # Create the Method:
    method = ExampleA2CMethod(render=True)

    # Apply the Method onto the Setting to get Results.
    results = setting.apply(method)
    print(results.summary())

    # BONUS: Running a hyper-parameter sweep:
    # method.hparam_sweep(setting)
