""" Utility functions for determining the observation space for a given SL dataset.
"""
from typing import Any, Dict, List, Optional, Sequence

import gym
import numpy as np
import torch
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
)
from continuum.tasks import TaskSet
from gym import Space, spaces
from torch.utils.data import Subset, TensorDataset

from sequoia.common.spaces import ImageTensorSpace, TensorBox, TensorDiscrete
from sequoia.common.spaces.image import could_become_image
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)


base_observation_spaces: Dict[str, Space] = {
    dataset_class.__name__.lower(): space
    for dataset_class, space in {
        MNIST: ImageTensorSpace(0, 1, shape=(1, 28, 28)),
        FashionMNIST: ImageTensorSpace(0, 1, shape=(1, 28, 28)),
        KMNIST: ImageTensorSpace(0, 1, shape=(1, 28, 28)),
        EMNIST: ImageTensorSpace(0, 1, shape=(1, 28, 28)),
        QMNIST: ImageTensorSpace(0, 1, shape=(1, 28, 28)),
        MNISTFellowship: ImageTensorSpace(0, 1, shape=(1, 28, 28)),
        # TODO: Determine the true bounds on the image values in cifar10.
        # Appears to be  ~= [-2.5, 2.5]
        CIFAR10: ImageTensorSpace(-np.inf, np.inf, shape=(3, 32, 32)),
        CIFAR100: ImageTensorSpace(-np.inf, np.inf, shape=(3, 32, 32)),
        CIFARFellowship: ImageTensorSpace(-np.inf, np.inf, shape=(3, 32, 32)),
        ImageNet100: ImageTensorSpace(0, 1, shape=(224, 224, 3)),
        ImageNet1000: ImageTensorSpace(0, 1, shape=(224, 224, 3)),
        Core50: ImageTensorSpace(0, 1, shape=(224, 224, 3)),
        Core50v2_79: ImageTensorSpace(0, 1, shape=(224, 224, 3)),
        Core50v2_196: ImageTensorSpace(0, 1, shape=(224, 224, 3)),
        Core50v2_391: ImageTensorSpace(0, 1, shape=(224, 224, 3)),
        Synbols: ImageTensorSpace(0, 1, shape=(3, 32, 32)),
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

CTRL_INSTALLED: bool = False
CTRL_STREAMS: List[str] = []
CTRL_NB_TASKS: Dict[str, Optional[int]] = {}
try:
    from ctrl.tasks.task import Task
    from ctrl.tasks.task_generator import TaskGenerator
except ImportError as exc:
    logger.debug(f"ctrl-bench isn't installed: {exc}")
    # Creating those just for type hinting.
    class Task:
        pass

    class TaskGenerator:
        pass

else:
    CTRL_INSTALLED = True
    CTRL_STREAMS = ["s_plus", "s_minus", "s_in", "s_out", "s_pl", "s_long"]
    n_tasks = [5, 5, 5, 5, 4, None]
    CTRL_NB_TASKS = dict(zip(CTRL_STREAMS, n_tasks))
    x_dims = [(3, 32, 32)] * len(CTRL_STREAMS)
    n_classes = [10, 10, 10, 10, 10, 5]

    for i, stream_name in enumerate(CTRL_STREAMS):
        # Create the 'base observation space' for this stream.
        obs_space = ImageTensorSpace(0, 1, shape=x_dims[i], dtype=torch.float32)

        # TODO: Not sure if the classes should be considered 'shared' or 'distinct'.
        # For now assume they are shared, so the setting's action space is always [0, 5]
        # but the action changes.
        # total_n_classes = n_tasks[i] * n_classes[i]
        # action_space = TensorDiscrete(n=total_n_classes)
        n_classes_per_task = n_classes[i]
        action_space = TensorDiscrete(n=n_classes_per_task)

        base_observation_spaces[stream_name] = obs_space
        base_action_spaces[stream_name] = action_space


from functools import singledispatch


@singledispatch
def get_observation_space(dataset: Any) -> gym.Space:
    raise NotImplementedError(
        f"Don't yet have a registered handler to get the observation space of dataset "
        f"{dataset}."
    )


@get_observation_space.register(Subset)
def _get_observation_space_for_subset(dataset: Subset) -> gym.Space:
    # The observations space of a Subset dataset is actually the same as the original
    # dataset.
    return get_observation_space(dataset.dataset)


@get_observation_space.register(str)
def _get_observation_space_for_dataset_name(dataset: str) -> gym.Space:
    if dataset not in base_observation_spaces:
        raise NotImplementedError(
            f"Can't yet tell what the 'base' observation space is for dataset "
            f"{dataset} because it doesn't have an entry in the "
            f"`base_observation_spaces` dict."
        )
    return base_observation_spaces[dataset]


@get_observation_space.register(TaskSet)
def _get_observation_space_for_taskset(dataset: TaskSet) -> gym.Space:
    assert False, dataset
    # return get_observation_space(type(dataset).__name__.lower())


@get_observation_space.register(TensorDataset)
def _get_observation_space_for_tensor_dataset(dataset: TensorDataset) -> gym.Space:
    x = dataset.tensors[0]
    if not (1 <= len(dataset.tensors) <= 2) or not (2 <= x.dim()):
        raise NotImplementedError(
            f"For now, can only handle TensorDatasets with 1 or 2 tensors. (x and y) "
            f"but dataset {dataset} has {len(dataset.tensors)}!"
        )

    low = x.min().cpu().item()
    high = x.max().cpu().item()
    obs_space = TensorBox(low=low, high=high, shape=x.shape[1:], dtype=x.dtype)
    if could_become_image(obs_space):
        obs_space = ImageTensorSpace.wrap(obs_space)
    return obs_space


@singledispatch
def get_action_space(dataset: Any) -> gym.Space:
    raise NotImplementedError(
        f"Don't yet have a registered handler to get the action space of dataset " f"{dataset}."
    )


@get_action_space.register(Subset)
def _get_action_space_for_subset(dataset: Subset) -> gym.Space:
    # The actions space of a Subset dataset is actually the same as the original
    # dataset.
    return get_action_space(dataset.dataset)


@get_action_space.register(str)
def _get_action_space_for_dataset_name(dataset: str) -> gym.Space:
    if dataset not in base_action_spaces:
        raise NotImplementedError(
            f"Can't yet tell what the 'base' action space is for dataset "
            f"{dataset} because it doesn't have an entry in the "
            f"`base_action_spaces` dict."
        )
    return base_action_spaces[dataset]


@singledispatch
def get_reward_space(dataset: Any) -> gym.Space:
    raise NotImplementedError(
        f"Don't yet have a registered handler to get the reward space of dataset " f"{dataset}."
    )


@get_reward_space.register(Subset)
def _get_reward_space_for_subset(dataset: Subset) -> gym.Space:
    # The rewards space of a Subset dataset is *usually* the same as the original
    # dataset.
    # TODO: Need to check this though? Maybe we're taking only the indices with a given class
    return get_reward_space(dataset.dataset)


@get_reward_space.register(str)
def _get_reward_space_for_dataset_name(dataset: str) -> gym.Space:
    if dataset not in base_reward_spaces:
        raise NotImplementedError(
            f"Can't yet tell what the 'base' reward space is for dataset "
            f"{dataset} because it doesn't have an entry in the "
            f"`base_reward_spaces` dict."
        )
    return base_reward_spaces[dataset]


@get_reward_space.register(TensorDataset)
@get_action_space.register(TensorDataset)
def get_y_space_for_tensor_dataset(dataset: TensorDataset) -> gym.Space:
    if len(dataset.tensors) != 2:
        raise NotImplementedError(
            f"Only able to detect the action space of TensorDatasets if they have two "
            f"tensors for now (x and y), but dataset {dataset} has {len(dataset.tensors)}!"
        )
    y = dataset.tensors[-1]
    low = y.min().item()
    high = y.max().item()
    y_sample_shape = y.shape[1:]

    if y.dtype.is_floating_point:
        return TensorBox(low, high, shape=y_sample_shape, dtype=y.dtype)

    # Integer y:
    if low == 0:
        n_classes = high + 1
        return TensorDiscrete(n_classes)

    # TODO: Add a space like DiscreteWithOffset ?
    return TensorBox(low, high, shape=y_sample_shape, dtype=y.dtype)


@get_action_space.register(list)
@get_action_space.register(tuple)
def _get_action_space_for_list_of_datasets(datasets: Sequence[TaskSet]) -> gym.Space:
    # TODO: IDEA: If given a list of datasets, try to find the 'union' of their spaces.
    # This is meant to be one potential solution to the case where custom datasets are
    # passed for each task, like [0, 2), [3, 4], etc.
    action_spaces = [get_action_space(dataset) for dataset in datasets]
    if isinstance(action_spaces[0], spaces.Discrete):
        lows = [0 if isinstance(space, spaces.Discrete) else space.low for space in action_spaces]
        highs = [
            space.n - 1 if isinstance(space, spaces.Discrete) else space.high
            for space in action_spaces
        ]

    if isinstance(reward_spaces[0], spaces.Discrete) and min(lows) == 0:
        return TensorDiscrete(max(highs) + 1)

    raise NotImplementedError(
        f"Don't yet know how to get the 'union' of the action spaces ({action_spaces}) "
        f" of datasets {datasets}"
    )


@get_reward_space.register(list)
@get_reward_space.register(tuple)
def _get_reward_space_for_list_of_datasets(datasets: Sequence[TaskSet]) -> gym.Space:
    # TODO: IDEA: If given a list of datasets, try to find the 'union' of their spaces.
    # This is meant to be one potential solution to the case where custom datasets are
    # passed for each task, like [0, 2), [3, 4], etc.
    reward_spaces = [get_reward_space(dataset) for dataset in datasets]
    if isinstance(reward_spaces[0], spaces.Discrete):
        lows = [0 if isinstance(space, spaces.Discrete) else space.low for space in reward_spaces]
        highs = [
            space.n - 1 if isinstance(space, spaces.Discrete) else space.high
            for space in reward_spaces
        ]

    if isinstance(reward_spaces[0], spaces.Discrete) and min(lows) == 0:
        return TensorDiscrete(max(highs) + 1)

    raise NotImplementedError(
        f"Don't yet know how to get the 'union' of the reward spaces ({reward_spaces}) "
        f" of datasets {datasets}"
    )
