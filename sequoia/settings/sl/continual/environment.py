""" WIP: Continual SL environment. (smooth task boundaries, etc)
"""
from typing import Any, Callable, Dict, List, Tuple, Type, Union, Sequence, Optional
from functools import partial
import gym
import numpy as np
from continuum.datasets import (
    CIFAR10,
    CIFAR100,
    EMNIST,
    KMNIST,
    MNIST,
    QMNIST,
    CIFARFellowship,
    Core50,
    Core50v2_79,
    Core50v2_196,
    Core50v2_391,
    FashionMNIST,
    ImageNet100,
    ImageNet1000,
    MNISTFellowship,
    Synbols,
    _ContinuumDataset,
)
from gym import Space, spaces
from sequoia.common.gym_wrappers.convert_tensors import (
    add_tensor_support as tensor_space,
)
import itertools
from sequoia.common.spaces import Image, NamedTupleSpace
from sequoia.settings.sl.environment import PassiveEnvironment
from sequoia.utils.logging_utils import get_logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
from sequoia.common.transforms import Transforms
from sequoia.common.gym_wrappers.batch_env.tile_images import tile_images
from .objects import (
    Actions,
    ActionType,
    Observations,
    ObservationType,
    Rewards,
    RewardType,
)

logger = get_logger(__file__)


base_observation_spaces: Dict[str, Space] = {
    dataset_class.__name__.lower(): space
    for dataset_class, space in {
        MNIST: tensor_space(Image(0, 1, shape=(1, 28, 28))),
        FashionMNIST: tensor_space(Image(0, 1, shape=(1, 28, 28))),
        KMNIST: tensor_space(Image(0, 1, shape=(1, 28, 28))),
        EMNIST: tensor_space(Image(0, 1, shape=(1, 28, 28))),
        QMNIST: tensor_space(Image(0, 1, shape=(1, 28, 28))),
        MNISTFellowship: tensor_space(Image(0, 1, shape=(1, 28, 28))),
        # TODO: Determine the true bounds on the image values in cifar10.
        # Appears to be  ~= [-2.5, 2.5]
        CIFAR10: tensor_space(Image(-np.inf, np.inf, shape=(3, 32, 32))),
        CIFAR100: tensor_space(Image(-np.inf, np.inf, shape=(3, 32, 32))),
        CIFARFellowship: tensor_space(Image(-np.inf, np.inf, shape=(3, 32, 32))),
        ImageNet100: tensor_space(Image(0, 1, shape=(224, 224, 3))),
        ImageNet1000: tensor_space(Image(0, 1, shape=(224, 224, 3))),
        Core50: tensor_space(Image(0, 1, shape=(224, 224, 3))),
        Core50v2_79: tensor_space(Image(0, 1, shape=(224, 224, 3))),
        Core50v2_196: tensor_space(Image(0, 1, shape=(224, 224, 3))),
        Core50v2_391: tensor_space(Image(0, 1, shape=(224, 224, 3))),
        Synbols: tensor_space(Image(0, 1, shape=(3, 32, 32))),
    }.items()
}


base_action_spaces: Dict[str, Space] = {
    dataset_class.__name__.lower(): space
    for dataset_class, space in {
        MNIST: spaces.Discrete(10),
        FashionMNIST: spaces.Discrete(10),
        KMNIST: spaces.Discrete(10),
        EMNIST: spaces.Discrete(10),
        QMNIST: spaces.Discrete(10),
        MNISTFellowship: spaces.Discrete(30),
        CIFAR10: spaces.Discrete(10),
        CIFAR100: spaces.Discrete(100),
        CIFARFellowship: spaces.Discrete(110),
        ImageNet100: spaces.Discrete(100),
        ImageNet1000: spaces.Discrete(1000),
        Core50: spaces.Discrete(50),
        Core50v2_79: spaces.Discrete(50),
        Core50v2_196: spaces.Discrete(50),
        Core50v2_391: spaces.Discrete(50),
        Synbols: spaces.Discrete(48),
    }.items()
}

# NOTE: Since the current SL datasets are image classification, the reward spaces are
# the same as the action space. But that won't be the case when we add other types of
# datasets!
base_reward_spaces: Dict[str, Space] = {
    dataset_name: action_space
    for dataset_name, action_space in base_action_spaces.items()
    if isinstance(action_space, spaces.Discrete)
}


def split_batch(
    batch: Tuple[Tensor, ...],
    hide_task_labels: bool,
    Observations=Observations,
    Rewards=Rewards,
) -> Tuple[Observations, Rewards]:
    """Splits the batch into a tuple of Observations and Rewards.

    Parameters
    ----------
    batch : Tuple[Tensor, ...]
        A batch of data coming from the dataset.

    Returns
    -------
    Tuple[Observations, Rewards]
        A tuple of Observations and Rewards.
    """
    # In this context (class_incremental), we will always have 3 items per
    # batch, because we use the ClassIncremental scenario from Continuum.
    assert len(batch) == 3
    x, y, t = batch

    if hide_task_labels:
        # Remove the task labels if we're not currently allowed to have
        # them.
        # TODO: Using None might cause some issues. Maybe set -1 instead?
        t = None

    observations = Observations(x=x, task_labels=t)
    rewards = Rewards(y=y)
    return observations, rewards


# IDEA: Have this env be the 'wrapper' / base env type for the continual SL envs, and
# register them in gym!
def default_split_batch_function(
    hide_task_labels: bool,
    Observations: Type[ObservationType] = Observations,
    Rewards: Type[RewardType] = Rewards,
) -> Callable[[Tuple[Tensor, ...]], Tuple[ObservationType, RewardType]]:
    """ Returns a callable that is used to split a batch into observations and rewards.
    """
    return partial(
        split_batch,
        hide_task_labels=hide_task_labels,
        Observations=Observations,
        Rewards=Rewards,
    )


class ContinualSLEnvironment(
    PassiveEnvironment[ObservationType, ActionType, RewardType]
):
    """ Continual Supervised Learning Environment.

    TODO: Here we actually inform the environment of its observation / action / reward
    spaces, which isn't ideal, but is arguably better than giving the env the
    responsibility (and arguments needed) to create the datasets of each task for the
    right split, apply the transforms,
    of each task and to use
    the right train/val/test split   
    """

    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        hide_task_labels: bool = True,
        observation_space: NamedTupleSpace = None,
        action_space: gym.Space = None,
        reward_space: gym.Space = None,
        Observations: Type[ObservationType] = Observations,
        Actions: Type[ActionType] = Actions,
        Rewards: Type[RewardType] = Rewards,
        split_batch_fn: Callable[
            [Tuple[Any, ...]], Tuple[ObservationType, ActionType]
        ] = None,
        pretend_to_be_active: bool = False,
        strict: bool = False,
        one_epoch_only: bool = True,
        drop_last: bool = False,
        **kwargs,
    ):
        assert isinstance(dataset, Dataset)
        self._hide_task_labels = hide_task_labels
        split_batch_fn = default_split_batch_function(
            hide_task_labels=hide_task_labels,
            Observations=Observations,
            Rewards=Rewards,  # TODO: Fix this 'Rewards' being of the 'wrong' type.
        )
        self._one_epoch_only = one_epoch_only
        super().__init__(
            dataset=dataset,
            split_batch_fn=split_batch_fn,
            observation_space=observation_space,
            action_space=action_space,
            reward_space=reward_space,
            pretend_to_be_active=pretend_to_be_active,
            strict=strict,
            drop_last=drop_last,
            **kwargs,
        )
        # TODO: Clean up the batching of a Sparse(Discrete) space so its less ugly.

    def step(
        self, action: ActionType
    ) -> Tuple[ObservationType, Optional[RewardType], bool, Sequence[Dict]]:
        obs, reward, done, info = super().step(action)
        if done and self._one_epoch_only:
            self.close()
        return obs, reward, done, info

    def __iter__(self):
        yield from super().__iter__()
        if self._one_epoch_only:
            self.close()

    # TODO: Remove / fix this 'split batch function'. The problem is that we need to
    # tell the environment how to take the three items from continuum and convert them
    # into


from pathlib import Path
from typing import Optional

import torch
from sequoia.common.config import Config
from sequoia.common.metrics import ClassificationMetrics, Metrics, MetricsType
from sequoia.settings.assumptions.continual import TestEnvironment
from sequoia.settings.assumptions.incremental_results import (
    TaskResults,
    TaskSequenceResults,
)
from sequoia.utils.logging_utils import get_logger

from .results import ContinualSLResults


class ContinualSLTestEnvironment(TestEnvironment[ContinualSLEnvironment]):
    def __init__(
        self,
        env: ContinualSLEnvironment,
        directory: Path,
        hide_task_labels: bool = True,
        step_limit: Optional[int] = None,
        no_rewards: bool = False,
        config: Config = None,
        **kwargs,
    ):
        super().__init__(
            env,
            directory=directory,
            step_limit=step_limit,
            no_rewards=no_rewards,
            config=config,
            **kwargs,
        )
        # IDEA: Make the env give us the task ids, and then hide them again after, just
        # so we can get propper 'per-task' metrics.
        # NOTE: This wouldn't be ideal however, as would assume that there is a 'discrete'
        # set of values for the task id, which is a
        assert isinstance(self.env.unwrapped, ContinualSLEnvironment)
        self.env.unwrapped.hide_task_labels = False

        self._steps = 0
        self.results: TaskResults = TaskResults()
        self._reset = False
        self.action_: Optional[ActionType] = None

    def get_results(self) -> ContinualSLResults:
        return self.results

    def reset(self):
        if not self._reset:
            logger.debug("Initial reset.")
            self._reset = True
            return super().reset()
        else:
            logger.debug("Resetting the env closes it.")
            self.close()
            return None

    def _before_step(self, action):
        self.action_ = action
        return super()._before_step(action)

    def _after_step(self, observation, reward, done, info):

        assert isinstance(reward, Tensor)
        action = self.action_
        assert action is not None
        actions = torch.as_tensor(action)

        batch_size = reward.shape[0]
        assert isinstance(
            self.action_space, (spaces.MultiDiscrete, spaces.MultiBinary)
        ), (
            f"TODO: Remove the assumption here that the env is a classification env "
            f"({self.action_space})"
        )
        # TODO: Switch this out for a generic function.
        fake_logits = torch.zeros([batch_size, self.action_space.nvec[0]], dtype=int)
        # FIXME: There must be a smarter way to do this indexing.
        for i, action in enumerate(actions):
            fake_logits[i, action] = 1
        actions = fake_logits
        metric = ClassificationMetrics(y=reward, y_pred=actions)
        reward = metric.accuracy

        self.results.metrics.append(metric)
        self._steps += 1

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

        # Record stats
        self.stats_recorder.after_step(observation, reward, done, info)

        # Record video
        if self.config.render:
            self.video_recorder.capture_frame()
        return done
        ##

    def _after_reset(self, observation: ObservationType):
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

