from typing import Iterable, Tuple, Callable

import gym
import pytest
import torch
from gym.wrappers import ClipAction
from gym.wrappers.pixel_observation import PixelObservationWrapper
from torch import Tensor
from torch.utils.data import TensorDataset

from sequoia.common.spaces import TensorBox, TensorDiscrete, TypedDictSpace
from sequoia.settings.sl import SLEnvironment
from sequoia.settings.sl.continual import Actions, Observations, Rewards

from .pixel_observation import PixelObservationWrapper
from .utils import has_wrapper, IterableWrapper
from .transform_wrappers import TransformReward


@pytest.mark.parametrize(
    "env,wrapper_type,result",
    [
        (lambda: PixelObservationWrapper(gym.make("CartPole-v0")), ClipAction, False),
        (
            lambda: PixelObservationWrapper(gym.make("CartPole-v0")),
            PixelObservationWrapper,
            True,
        ),
        (
            lambda: PixelObservationWrapper(gym.make("CartPole-v0")),
            PixelObservationWrapper,
            True,
        ),
        # param_requires_atari_py(AtariPreprocessing(gym.make("Breakout-v0")), ClipAction, True),
    ],
)
def test_has_wrapper(env, wrapper_type, result):
    assert has_wrapper(env(), wrapper_type) == result

