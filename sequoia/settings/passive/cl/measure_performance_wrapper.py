""" TODO: Create a Wrapper that measures performance over the first epoch of training in SL.

Then maybe after we can make something more general that also works for RL.
"""
import warnings
from abc import ABC
from collections import defaultdict
from typing import Dict, Generic, Iterable, List, Optional, Tuple, Union, Sequence, Any

import wandb
import numpy as np
from gym.utils import colorize
from torch import Tensor
from gym.vector import VectorEnv
from sequoia.common.gym_wrappers.utils import IterableWrapper, EnvType
from sequoia.common.metrics import ClassificationMetrics, Metrics, MetricsType
from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.settings.base import (
    Actions,
    Environment,
    Observations,
    Rewards,
)
from sequoia.utils.utils import add_prefix
from sequoia.settings.passive.passive_environment import PassiveEnvironment


class MeasurePerformanceWrapper(
    IterableWrapper[EnvType], Generic[EnvType, MetricsType], ABC
):
    def __init__(self, env: Environment):
        super().__init__(env)
        self._metrics: Dict[int, MetricsType] = {}

    def get_online_performance(self) -> Dict[int, List[MetricsType]]:
        """Returns the online performance over the evaluation period.

        Returns
        -------
        Dict[int, MetricsType]
            A dict mapping from step number to the Metrics object captured at that step.
        """
        return dict(self._metrics.copy())

    def get_average_online_performance(self) -> Optional[MetricsType]:
        """Returns the average online performance over the evaluation period, or None
        if the env was not iterated over / interacted with.

        Returns
        -------
        Optional[MetricsType]
            Metrics
        """
        if not self._metrics:
            return None
        return sum(self._metrics.values())


class MeasureSLPerformanceWrapper(
    MeasurePerformanceWrapper[PassiveEnvironment, ClassificationMetrics]
):
    def __init__(
        self,
        env: PassiveEnvironment,
        first_epoch_only: bool = False,
        wandb_prefix: str = None,
    ):
        super().__init__(env)
        self._metrics: Dict[int, ClassificationMetrics] = defaultdict(Metrics)
        self.first_epoch_only = first_epoch_only
        self.wandb_prefix = wandb_prefix
        # Counter for the number of steps.
        self._steps: int = 0
        assert isinstance(self.env.unwrapped, PassiveEnvironment)
        if not self.env.unwrapped.pretend_to_be_active:
            warnings.warn(
                RuntimeWarning(
                    colorize(
                        "Your performance "
                        + ("during the first epoch " if self.first_epoch_only else "")
                        + "on this environment will be monitored! "
                        "Since this env is Passive, i.e. a Supervised Learning "
                        "DataLoader, the Rewards (y) will be withheld until "
                        "actions are passed to the 'send' method. Make sure that "
                        "your training loop can handle this small tweak.",
                        color="yellow",
                    )
                )
            )
        self.env.unwrapped.pretend_to_be_active = True
        self.__epochs = 0

    def reset(self) -> Observations:
        return self.env.reset()

    @property
    def in_evaluation_period(self) -> bool:
        if self.first_epoch_only:
            # TODO: Double-check the iteraction of IterableDataset and __len__
            return self.__epochs == 0
        return True

    def step(self, action: Actions):
        observation, reward, done, info = self.env.step(action)
        if self.in_evaluation_period:
            self._metrics[self._steps] += self.get_metrics(action, reward)
        self._steps += 1
        return observation, reward, done, info

    def send(self, action: Actions):
        if not isinstance(action, Actions):
            assert isinstance(action, (np.ndarray, Tensor))
            action = Actions(action)

        reward = self.env.send(action)

        if self.in_evaluation_period:
            # TODO: Edge case, but we also need the prediction for the last batch to be
            # counted.
            self._metrics[self._steps] += self.get_metrics(action, reward)
        elif self.first_epoch_only:
            # If we are at the last batch in the first epoch, we still keep the metrics
            # for that batch, even though we're technically not in the first epoch
            # anymore.
            there_is_last_batch = self.env.__len__() % self.batch_size != 0
            last_batch_isnt_dropped = not self.env.unwrapped.drop_last
            currently_at_last_batch = self._steps == self.env.__len__() - 1
            if (
                there_is_last_batch
                and last_batch_isnt_dropped
                and currently_at_last_batch
            ):
                self._metrics[self._steps] += self.get_metrics(action, reward)

        # This is ok since we don't increment in the iterator.
        self._steps += 1
        return reward

    def get_metrics(self, action: Actions, reward: Rewards) -> Metrics:
        assert action.y_pred.shape == reward.y.shape, (action.shapes, reward.shapes)
        metric = ClassificationMetrics(
            y_pred=action.y_pred, y=reward.y, num_classes=self.n_classes
        )

        if wandb.run:
            log_dict = metric.to_log_dict()
            if self.wandb_prefix:
                log_dict = add_prefix(log_dict, prefix=self.wandb_prefix, sep="/")
            log_dict["steps"] = self._steps
            wandb.log(log_dict)
        return metric

    def __iter__(self) -> Iterable[Tuple[Observations, Optional[Rewards]]]:
        if self.__epochs == 1 and self.first_epoch_only:
            print(
                colorize(
                    "Your performance during the first epoch on this environment has "
                    "been successfully measured! The environment will now yield the "
                    "rewards (y) during iteration, and you are no longer required to "
                    "send an action for each observation.",
                    color="green",
                )
            )
            self.env.unwrapped.pretend_to_be_active = False

        for obs, rew in self.env.__iter__():
            if self.in_evaluation_period:
                yield obs, None
            else:
                yield obs, rew
        self.__epochs += 1


from sequoia.settings.active import ActiveEnvironment


class MeasureRLPerformanceWrapper(
    MeasurePerformanceWrapper[ActiveEnvironment, EpisodeMetrics]
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

        self.is_batched_env = isinstance(self.env.unwrapped, VectorEnv)
        self._batch_size = self.env.num_envs if self.is_batched_env else 1

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
        obs = self.env.reset()
        # assert isinstance(obs, Observations)
        return obs

    def step(self, action: Actions):
        observation, rewards_, done, info = self.env.step(action)
        self._steps += 1
        reward = rewards_.y if isinstance(rewards_, Rewards) else rewards_

        if isinstance(done, bool):
            self._episodes += int(done)
        elif isinstance(done, np.ndarray):
            self._episodes += sum(done)
        else:
            self._episodes += done.int().sum()

        if self.in_evaluation_period:
            if self.is_batched_env:
                for env_index, (env_is_done, env_reward) in enumerate(
                    zip(done, reward)
                ):
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

    def send(self, action: Actions) -> Rewards:
        rewards_ = self.env.send(action)
        self._steps += 1
        reward = rewards_.y if isinstance(rewards_, Rewards) else rewards_

        # TODO: Need access to the "done" signal in here somehow.
        done = self.env.done_

        if isinstance(done, bool):
            self._episodes += int(done)
        elif isinstance(done, np.ndarray):
            self._episodes += sum(done)
        else:
            self._episodes += done.int().sum()

        if self.in_evaluation_period:
            if self.is_batched_env:
                for env_index, (env_is_done, env_reward) in enumerate(
                    zip(done, reward)
                ):
                    self._current_episode_reward[env_index] += env_reward
                    self._current_episode_steps[env_index] += 1
            else:
                self._current_episode_reward[0] += reward
                self._current_episode_steps[0] += 1

            metrics = self.get_metrics(action, reward, done)

            if metrics is not None:
                assert self._steps not in self._metrics, "two metrics at same step?"
                self._metrics[self._steps] = metrics

        return rewards_

    def get_metrics(
        self,
        action: Union[Actions, Any],
        reward: Union[Rewards, Any],
        done: Union[bool, Sequence[bool]],
    ) -> Optional[EpisodeMetrics]:
        metrics = []

        rewards = reward.y if isinstance(reward, Rewards) else reward
        actions = action.y_pred if isinstance(action, Actions) else action
        dones: Sequence[bool]
        if not self.is_batched_env:
            rewards = [rewards]
            actions = [actions]
            assert isinstance(done, bool)
            dones = [done]
        else:
            assert isinstance(done, (np.ndarray, Tensor))
            dones = done

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
