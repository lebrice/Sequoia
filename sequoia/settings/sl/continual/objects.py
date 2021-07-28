from dataclasses import dataclass
from sequoia.settings.sl import SLSetting
from torch import Tensor
from typing import TypeVar, Optional
from sequoia.settings.assumptions.continual import ContinualAssumption
from sequoia.common.spaces import Image, Sparse, TypedDictSpace, ImageTensorSpace
from gym import spaces


@dataclass(frozen=True)
class Observations(SLSetting.Observations, ContinualAssumption.Observations):
    """ Observations from a Continual Supervised Learning environment. """
    x: Tensor
    task_labels: Optional[Tensor] = None


ObservationType = TypeVar("ObservationType", bound=Observations)
import torch

class ObservationSpace(TypedDictSpace[ObservationType]):
    """ Observation space of a Continual SL Setting. """
    # The sample space: this is a gym.spaces.Box subclass with added properties for
    # images, such as `channels`, `h`, `w`, `is_channels_first`, etc.
    # This space will return Tensors.
    x: ImageTensorSpace
    # The task label space: This is a gym.spaces.MultiDiscrete of Tensors.
    task_labels: Sparse[torch.LongTensor]


# TODO: Eventually also use some kind of structured action and reward space!
# TODO: Figure out how/where to switch the actions type to be specific to classification
# from sequoia.settings.assumptions.task_type import ClassificationActions


@dataclass(frozen=True)
class Actions(SLSetting.Actions):
    """ Actions to be sent to a Continual Supervised Learning environment. """
    y_pred: Tensor




class ActionSpace(TypedDictSpace):
    """ Action space of a Continual SL Setting. """
    y_pred: spaces.Space


@dataclass(frozen=True)
class Rewards(SLSetting.Rewards):
    """ Rewards obtained from a Continual Supervised Learning environment. """
    y: Tensor


class RewardSpace(TypedDictSpace):
    """ Reward space of a Continual SL Setting. """
    y: spaces.Space




ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)
