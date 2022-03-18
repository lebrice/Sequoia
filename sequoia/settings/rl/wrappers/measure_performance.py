""" TODO: Create a Wrapper that measures performance over the first epoch of training in SL.

Then maybe after we can make something more general that also works for RL.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from torch import Tensor

import wandb
from sequoia.common.gym_wrappers.measure_performance import MeasurePerformanceWrapper
from sequoia.common.metrics import Metrics
from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.settings.base import Actions, Observations, Rewards
from sequoia.settings.rl import ActiveEnvironment
from sequoia.utils.utils import add_prefix


class MeasureRLPerformanceWrapper(
    MeasurePerformanceWrapper
    # MeasurePerformanceWrapper[ActiveEnvironment]  # python 3.7
    # MeasurePerformanceWrapper[ActiveEnvironment, EpisodeMetrics] # python 3.8+
):
    def __init__(
        self,
        env: ActiveEnvironment,
        eval_episodes: int = None,
        eval_steps: int = None,
        wandb_prefix: str = None,
    ):
        super().__init__(env)
        self._metrics: Dict[int, EpisodeMetrics] = {}
        self._eval_episodes = eval_episodes or 0
        self._eval_steps = eval_steps or 0
        # Counter for the number of steps.
        self._steps: int = 0
        # Counter for the number of episodes
        self._episodes: int = 0
        self.wandb_prefix = wandb_prefix

        self._batch_size = self.env.num_envs if self.is_vectorized else 1

        self._current_episode_reward = np.zeros([self._batch_size], dtype=float)
        self._current_episode_steps = np.zeros([self._batch_size], dtype=int)

    @property
    def in_evaluation_period(self) -> bool:
        """Returns wether the performance is currently being monitored.

        Returns
        -------
        bool
            Wether or not the performance on the env is being monitored.
        """
        if self._eval_steps:
            return self._steps <= self._eval_steps
        if self._eval_episodes:
            return self._eval_episodes <= self._eval_episodes
        return True

    def reset(self) -> Union[Observations, Any]:
        obs = super().reset()
        # assert isinstance(obs, Observations)
        return obs

    def step(self, action: Actions):
        observation, rewards_, done, info = super().step(action)
        self._steps += 1
        reward = rewards_.y if isinstance(rewards_, Rewards) else rewards_

        if isinstance(done, bool):
            self._episodes += int(done)
        elif isinstance(done, np.ndarray):
            self._episodes += sum(done)
        else:
            self._episodes += done.int().sum()

        if self.in_evaluation_period:
            if self.is_vectorized:
                for env_index, (env_is_done, env_reward) in enumerate(zip(done, reward)):
                    self._current_episode_reward[env_index] += env_reward
                    self._current_episode_steps[env_index] += 1
            else:
                self._current_episode_reward[0] += reward
                self._current_episode_steps[0] += 1

            metrics = self.get_metrics(action, reward, done)

            if metrics is not None:
                assert self._steps not in self._metrics, "two metrics at same step?"
                self._metrics[self._steps] = metrics

        return observation, rewards_, done, info

    # def send(self, action: Actions) -> Rewards:
    # self.action_ = action
    # rewards_ = super().send(action)
    # self._steps += 1
    # reward = rewards_.y if isinstance(rewards_, Rewards) else rewards_

    # # TODO: Need access to the "done" signal in here somehow.
    # done = self.done_

    # if isinstance(done, bool):
    #     self._episodes += int(done)
    # elif isinstance(done, np.ndarray):
    #     self._episodes += sum(done)
    # else:
    #     self._episodes += done.int().sum()

    # if self.in_evaluation_period:
    #     if self.is_vectorized:
    #         for env_index, (env_is_done, env_reward) in enumerate(
    #             zip(done, reward)
    #         ):
    #             self._current_episode_reward[env_index] += env_reward
    #             self._current_episode_steps[env_index] += 1
    #     else:
    #         self._current_episode_reward[0] += reward
    #         self._current_episode_steps[0] += 1

    #     metrics = self.get_metrics(action, reward, done)

    #     if metrics is not None:
    #         assert self._steps not in self._metrics, "two metrics at same step?"
    #         self._metrics[self._steps] = metrics

    # return rewards_

    def get_metrics(
        self,
        action: Union[Actions, Any],
        reward: Union[Rewards, Any],
        done: Union[bool, Sequence[bool]],
    ) -> Optional[EpisodeMetrics]:
        # TODO: Add some metric about the entropy of the policy's distribution?
        rewards = reward.y if isinstance(reward, Rewards) else reward
        actions = action.y_pred if isinstance(action, Actions) else action
        dones: Sequence[bool]
        if not self.is_vectorized:
            rewards = [rewards]
            actions = [actions]
            assert isinstance(done, bool)
            dones = [done]
        else:
            assert isinstance(done, (np.ndarray, Tensor))
            dones = done

        metrics: List[EpisodeMetrics] = []
        for env_index, (env_is_done, reward) in enumerate(zip(dones, rewards)):
            if env_is_done:
                metrics.append(
                    EpisodeMetrics(
                        n_samples=1,
                        # The average reward per episode.
                        mean_episode_reward=self._current_episode_reward[env_index],
                        # The average length of each episode.
                        mean_episode_length=self._current_episode_steps[env_index],
                    )
                )
                self._current_episode_reward[env_index] = 0
                self._current_episode_steps[env_index] = 0

        if not metrics:
            return None

        metric = sum(metrics, Metrics())
        if wandb.run:
            log_dict = metric.to_log_dict()
            if self.wandb_prefix:
                log_dict = add_prefix(log_dict, prefix=self.wandb_prefix, sep="/")
            log_dict["steps"] = self._steps
            log_dict["episode"] = self._episodes
            wandb.log(log_dict)

        return metric
