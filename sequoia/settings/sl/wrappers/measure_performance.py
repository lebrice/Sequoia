""" Wrapper that gets applied onto the environment in order to measure the online
training performance.
"""
import warnings
from abc import ABC
from collections import defaultdict

from typing import Any, Dict, Generic, Iterator, List, Optional, Sequence, Tuple, Union
from gym import spaces

import torch
import numpy as np
import wandb
from gym.utils import colorize
from gym.vector import VectorEnv
from sequoia.common.gym_wrappers.measure_performance import MeasurePerformanceWrapper
from sequoia.common.gym_wrappers.utils import EnvType, IterableWrapper
from sequoia.common.metrics import ClassificationMetrics, Metrics, MetricsType
from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.settings.base import Actions, Environment, Observations, Rewards
from sequoia.settings.sl.environment import PassiveEnvironment
from sequoia.utils.utils import add_prefix
from torch import Tensor
from sequoia.common.gym_wrappers.batch_env.tile_images import tile_images
from sequoia.settings.assumptions.task_type import ClassificationActions


class MeasureSLPerformanceWrapper(
    MeasurePerformanceWrapper,
    # MeasurePerformanceWrapper[PassiveEnvironment]  # Python 3.7
    # MeasurePerformanceWrapper[PassiveEnvironment, ClassificationMetrics] # Python 3.8+
):
    def __init__(
        self,
        env: PassiveEnvironment,
        first_epoch_only: bool = False,
        wandb_prefix: str = None,
    ):
        super().__init__(env)
        # Metrics mapping from step to the metrics at that step.
        self._metrics: Dict[int, ClassificationMetrics] = {}
        self.first_epoch_only = first_epoch_only
        self.wandb_prefix = wandb_prefix
        # Counter for the number of steps.
        self._steps: int = 0
        assert isinstance(self.env.unwrapped, PassiveEnvironment)
        if not self.env.unwrapped.pretend_to_be_active:
            warnings.warn(
                RuntimeWarning(
                    colorize(
                        "Your online performance "
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
        self.__epochs = -1

        # TODO: Remove this: only works for classification:
        self._n_classes = self._get_n_classes()
        self._generator = None
        self._reward: Optional[Rewards] = None
        self._action: Optional[Actions] = None

    def reset(self) -> Observations:
        # NOTE: This calls the `observation` hook.
        return super().reset()

    @property
    def in_evaluation_period(self) -> bool:
        if self.first_epoch_only:
            return self.__epochs == 0
        return True

    def step(self, action: Actions):
        # NOTE: This calls all of the the `observation`, `action`, `reward`, etc hooks.
        observation, reward, done, info = super().step(action)
        return observation, reward, done, info

    def observation(self, observation):
        return super().observation(observation)

    def action(self, action):
        if not isinstance(action, Actions):
            # TODO: Remove this conversion to Actions, and instead expect methods to
            # give back the proper kind of 'Actions'.
            assert isinstance(action, (np.ndarray, Tensor))
            action = Actions(action)
        action = super().action(action)
        self._action = action
        return action

    def reward(self, reward):
        reward = super().reward(reward)
        # TODO: Make this wrapper task-aware, using the task ids in this `observation`?
        if not self.in_evaluation_period:
            return reward

        assert self.unwrapped.pretend_to_be_active
        # TODO: Edge case, but we also need the prediction for the last batch to be
        # counted.
        assert self._action is not None
        assert self._steps not in self._metrics, (self._steps, self._metrics)
        self._metrics[self._steps] = self.get_metrics(self._action, reward)

        self._action = None
        self._reward = None
        self._steps += 1

        return reward

    def _get_n_classes(self) -> Optional[int]:
        if isinstance(self.action_space, spaces.Dict):
            y_pred_space = self.action_space["y_pred"]
            if isinstance(y_pred_space, spaces.MultiDiscrete):
                return y_pred_space.nvec[0]
            elif isinstance(y_pred_space, spaces.Discrete):
                return y_pred_space.n
        elif isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            if (self.action_space.nvec == self.action_space.nvec[0]).all():
                return self.action_space.nvec[0]
        return None

    def get_metrics(self, action: Actions, reward: Rewards) -> Metrics:
        # TODO: Make this a bit simpler: if we don't have `drop_last` and we're in the
        # last batch, then truncate the action to fit the length of the reward.
        if action.y_pred.shape != reward.y.shape:
            # BUG: Keep getting this weird error where the reward and action shapes
            # don't match when using the ReplayEnvWrapper with a MeasureSLPerformanceWrapper.
            assert False, (self.length(), self._steps, self.__epochs, action.shapes, reward.shapes)
            assert action.y_pred.shape == reward.y.shape, (action.shapes, reward.shapes, self._steps, self.__epochs)

        if not isinstance(action, ClassificationActions):
            logits: Tensor
            if isinstance(action.y_pred, np.ndarray):
                action = action.torch(device=reward.device)
            if not action.y_pred.is_floating_point():
                logits = torch.nn.functional.one_hot(action.y_pred, self._n_classes)
            else:
                # The action already is the logits:
                assert action.y_pred.shape[-1] == self._n_classes
                logits = action.y_pred

            action = ClassificationActions(y_pred=logits.argmax(-1), logits=logits)

        metric = ClassificationMetrics(
            y_pred=action.y_pred, logits=action.logits, y=reward.y, num_classes=self.n_classes
        )

        if wandb.run:
            log_dict = metric.to_log_dict()
            if self.wandb_prefix:
                log_dict = add_prefix(log_dict, prefix=self.wandb_prefix, sep="/")
            log_dict["steps"] = self._steps
            wandb.log(log_dict)
        return metric
    
    def __iter__(self) -> Iterator[Tuple[Observations, Optional[Rewards]]]:
        self.__epochs += 1
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
        self.env.unwrapped._reward_queue.clear()

        #     if self._generator is not None:
        #         self._generator.close()          
        #     self._generator = env_generator_loop(self)
        #     yield from self._generator

        # yield from super().__iter__()

        for obs, rew in super().__iter__():
            if self.in_evaluation_period:
                assert rew is None, rew
                yield obs, None
            else:
                yield obs, rew

    def send(self, action: Actions):
        # NOTE: This calls the `action` and the `reward` hooks.
        return super().send(action)
