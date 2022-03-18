from functools import partial, singledispatch
from itertools import accumulate
from typing import Any, Dict, List

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from continuum import TaskSet
from torch import Tensor

from sequoia.common.gym_wrappers import IterableWrapper


@singledispatch
def relabel(data: Any, mapping: Dict[int, int] = None) -> Any:
    """Relabels the given data (from a task) so they all share the same action space."""
    raise NotImplementedError(f"Don't know how to relabel {data} of type {type(data)}")


@relabel.register
def relabel_ndarray(y: np.ndarray, mapping: Dict[int, int] = None) -> np.ndarray:
    new_y = y.copy()
    mapping = mapping or {c: i for i, c in enumerate(np.unique(y))}
    for old_label, new_label in mapping.items():
        new_y[y == old_label] = new_label
    return new_y


@relabel.register
def relabel_tensor(y: Tensor, mapping: Dict[int, int] = None) -> Tensor:
    new_y = y.copy()
    mapping = mapping or {c: i for i, c in enumerate(torch.unique(y))}
    for old_label, new_label in mapping.items():
        new_y[y == old_label] = new_label
    return new_y


@relabel.register
def relabel_taskset(task_set: TaskSet, mapping: Dict[int, int] = None) -> TaskSet:
    mapping = mapping or {c: i for i, c in enumerate(task_set.get_classes())}
    old_y = task_set._y
    new_y = relabel(old_y, mapping=mapping)
    assert not task_set.target_trsf
    # TODO: Two options here: Either create a new 'y' array, OR add a target_trsf that
    # does the remapping. Not sure if there's a benefit in doing one vs the other atm.
    # NOTE: Choosing to replace the `y` to make sure that the concatenated datasets keep
    # the transformed y.
    new_taskset = replace_taskset_attributes(task_set, y=new_y)
    return new_taskset


from sequoia.utils.generic_functions.replace import replace


@replace.register
def replace_taskset_attributes(task_set: TaskSet, **kwargs) -> TaskSet:
    new_kwargs = dict(
        x=task_set._x,
        y=task_set._y,
        t=task_set._t,
        trsf=task_set.trsf,
        target_trsf=task_set.target_trsf,
        data_type=task_set.data_type,
        bounding_boxes=task_set.bounding_boxes,
    )
    new_kwargs.update(kwargs)
    return type(task_set)(**new_kwargs)


class SharedActionSpaceWrapper(IterableWrapper):
    # """ Wrapper that gets applied to a ContinualSLEnvironment
    def __init__(self, env: gym.Env, task_classes: List[int]):
        self.task_classes = task_classes
        super().__init__(env=env, f=partial(relabel, task_classes=self.task_classes))


from collections import Counter

from .environment import ContinualSLEnvironment
from .objects import ObservationType, RewardType


class ShowLabelDistributionWrapper(IterableWrapper[ContinualSLEnvironment]):
    """Wrapper around a SL environment that shows the distribution of the labels.

    Shows the distributions of the task labels, if applicable.
    """

    def __init__(self, env: ContinualSLEnvironment, env_name: str):
        super().__init__(env=env)
        self.env_name = env_name
        # IDEA: Could use bins for continuous values ?
        # IDEA: Also use a counter for the actions?
        self.counters: Dict[str, List[Counter]] = {
            "y": [],
            "t": [],
        }

    def observation(self, observation: ObservationType) -> ObservationType:
        t = observation.task_labels
        if t is None:
            t = [None] * observation.batch_size
        if isinstance(t, Tensor):
            t = t.cpu().numpy()
        t_count = Counter(t)
        self.counters["t"].append(t_count)
        return observation

    def reward(self, reward: RewardType) -> RewardType:
        y = reward.y.cpu().numpy()
        y_count = Counter(y)
        self.counters["y"].append(y_count)
        return reward

    def make_figure(self) -> plt.Figure:
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(len(self.counters))
        # total_length: int = sum(sum(counter.values()) for counter in self.y_counters)

        for i, (name, counters) in enumerate(self.counters.items()):
            # Values for the x axis are the number of samples seen so far for each
            # batch.
            x = list(accumulate(sum(counter.values()) for counter in counters))
            unique_values = list(sorted(set().union(*counters)))
            for label in unique_values:
                y = [counter.get(label) for counter in counters]
                axes[i].plot(x, y, label=f"{name}={label}")
            axes[i].legend()
            axes[i].set_title(f"{self.env_name} {name}")
            axes[i].set_xlabel("Batch index")
            axes[i].set_ylabel("Count in batch")

        fig.set_size_inches((6, 4), forward=False)
        fig.legend()
        return fig
