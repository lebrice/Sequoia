from typing import Any, Callable, Tuple, Union

import gym
from gym import spaces
from sequoia.common.spaces import TypedDictSpace
from sequoia.settings.sl.continual.environment import ContinualSLEnvironment
from torch.utils.data import Dataset, IterableDataset

from ..continual.environment import ContinualSLTestEnvironment
from .objects import (
    Actions,
    ActionType,
    Observations,
    ObservationType,
    Rewards,
    RewardType,
)
from sequoia.settings.base.objects import Rewards as BaseRewards
from sequoia.utils.logging_utils import get_logger
logger = get_logger(__file__)


class IncrementalSLEnvironment(ContinualSLEnvironment[ObservationType, ActionType, RewardType]):
    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        hide_task_labels: bool = True,
        observation_space: TypedDictSpace[ObservationType] = None,
        action_space: gym.Space = None,
        reward_space: gym.Space = None,
        split_batch_fn: Callable[
            [Tuple[Any, ...]], Tuple[ObservationType, ActionType]
        ] = None,
        pretend_to_be_active: bool = False,
        strict: bool = False,
        one_epoch_only: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset,
            hide_task_labels=hide_task_labels,
            observation_space=observation_space,
            action_space=action_space,
            reward_space=reward_space,
            split_batch_fn=split_batch_fn,
            pretend_to_be_active=pretend_to_be_active,
            strict=strict,
            one_epoch_only=one_epoch_only,
            **kwargs,
        )


from .results import IncrementalSLResults
from sequoia.settings.assumptions.incremental import (
    TaskResults, TaskSequenceResults, TestEnvironment
)
from typing import Dict, Any
from sequoia.common.metrics import ClassificationMetrics
import torch
import warnings
import bisect
from sequoia.common.gym_wrappers.utils import tile_images
import numpy as np
from torch.nn import functional as F
from sequoia.common.transforms import Transforms


class IncrementalSLTestEnvironment(ContinualSLTestEnvironment):
    def __init__(
        self, env: gym.Env, *args, task_schedule: Dict[int, Any] = None, **kwargs
    ):
        super().__init__(env, *args, **kwargs)
        self._steps = 0
        # TODO: Maybe rework this so we don't depend on the test phase being one task at
        # a time, instead store the test metrics in the task corresponding to the
        # task_label in the observations.
        # BUG: The problem is, right now we're depending on being passed the
        # 'task schedule', which we then use to get the task ids. This
        # is actually pretty bad, because if the class ordering was changed between
        # training and testing, then, this wouldn't actually report the correct results!
        self.task_schedule = task_schedule or {}
        self.task_steps = sorted(self.task_schedule.keys())
        self.results: TaskSequenceResults[ClassificationMetrics] = TaskSequenceResults(
            task_results=[TaskResults() for step in self.task_steps]
        )
        # self._reset = False
        # NOTE: The task schedule is already in terms of the number of batches.
        self.boundary_steps = [step for step in self.task_schedule.keys()]

    def get_results(self) -> IncrementalSLResults:
        return self.results

    def reset(self):
        return super().reset()
        # if not self._reset:
        #     logger.debug("Initial reset.")
        #     self._reset = True
        #     return super().reset()
        # else:
        #     logger.debug("Resetting the env closes it.")
        #     self.close()
        #     return None

    def _before_step(self, action):
        self._action = action
        return super()._before_step(action)

    def _after_step(self, observation, reward, done, info):
        if not isinstance(reward, BaseRewards):
            reward = BaseRewards(y=torch.as_tensor(reward))

        batch_size = reward.batch_size

        action = self._action
        assert action is not None

        if isinstance(self.action_space, (spaces.MultiDiscrete, spaces.MultiBinary)):
            n_classes = self.action_space.nvec[0]
            from sequoia.settings.assumptions.task_type import ClassificationActions

            if not isinstance(action, ClassificationActions):
                if isinstance(action, Actions):
                    y_pred = action.y_pred
                    # 'upgrade', creating some fake logits.
                else:
                    y_pred = torch.as_tensor(action)
                fake_logits = F.one_hot(y_pred, n_classes)
                action = ClassificationActions(y_pred=y_pred, logits=fake_logits)
        else:
            raise NotImplementedError(
                f"TODO: Remove the assumption here that the env is a classification env "
                f"({self.action_space}, {self.reward_space})"
            )

        if action.batch_size != reward.batch_size:
            warnings.warn(
                RuntimeWarning(
                    f"Truncating the action since its batch size {action.batch_size} "
                    f"is larger than the rewards': ({reward.batch_size})"
                )
            )
            action = action[:, :reward.batch_size]

        # TODO: Use some kind of generic `get_metrics(actions: Actions, rewards: Rewards)`
        # function instead.
        y = reward.y
        logits = action.logits
        y_pred = action.y_pred
        metric = ClassificationMetrics(y=y, logits=logits, y_pred=y_pred)
        reward = metric.accuracy

        task_steps = sorted(self.task_schedule.keys())
        assert 0 in task_steps, task_steps

        nb_tasks = len(task_steps)
        assert nb_tasks >= 1

        # Given the step, find the task id.
        task_id = bisect.bisect_right(task_steps, self._steps) - 1
        self.results.task_results[task_id].metrics.append(metric)

        self._steps += 1

        # FIXME: Temporary fix: TODO: Make sure this doesn't truncate the number of labels
        if self._steps == self.step_limit - 1:
            self.close()
            done = True

        # Debugging issue with Monitor class:
        # return super()._after_step(observation, reward, done, info)
        if not self.enabled:
            return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the
            # first one of the new episode
            if self.config.render:
                self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record stats: (TODO: accuracy serves as the 'reward'!)
        reward_for_stats = metric.accuracy
        self.stats_recorder.after_step(observation, reward_for_stats, done, info)

        # Record video
        if self.config and self.config.render:
            self.video_recorder.capture_frame()
        return done

    def _after_reset(self, observation: Observations):
        image_batch = observation.numpy().x
        # Need to create a single image with the right dtype for the Monitor
        # from gym to create gifs / videos with it.
        if self.batch_size:
            # Need to tile the image batch so it can be seen as a single image
            # by the Monitor.
            image_batch = tile_images(image_batch)

        image_batch = Transforms.channels_last_if_needed(image_batch)
        if image_batch.dtype == np.float32:
            assert (0 <= image_batch).all() and (image_batch <= 1).all()
            image_batch = (256 * image_batch).astype(np.uint8)

        assert image_batch.dtype == np.uint8
        # Debugging this issue here:
        # super()._after_reset(image_batch)

        # -- Code from Monitor
        if not self.enabled:
            return
        # Reset the stat count
        self.stats_recorder.after_reset(observation)
        if self.config.render:
            self.reset_video_recorder()

        # Bump *after* all reset activity has finished
        self.episode_id += 1

        self._flush()
        # --

    def render(self, mode="human", **kwargs):
        # NOTE: This doesn't get called, because the video recorder uses
        # self.env.render(), rather than self.render()
        # TODO: Render when the 'render' argument in config is set to True.
        image_batch = super().render(mode=mode, **kwargs)
        if mode == "rgb_array" and self.batch_size:
            image_batch = tile_images(image_batch)
        return image_batch
