from typing import Any, Callable, Tuple, Union

import gym
from gym import spaces
from sequoia.common.spaces import TypedDictSpace
from sequoia.settings.sl.continual.environment import ContinualSLEnvironment
from torch.utils.data import Dataset, IterableDataset

from .objects import (
    Actions,
    ActionType,
    Observations,
    ObservationType,
    Rewards,
    RewardType,
)


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
