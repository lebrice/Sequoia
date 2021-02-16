import gym
import pytest
from gym.wrappers import AtariPreprocessing, ClipAction, FilterObservation
from gym.wrappers.pixel_observation import PixelObservationWrapper
from sequoia.conftest import param_requires_atari_py

from .pixel_observation import PixelObservationWrapper
from .utils import has_wrapper


@pytest.mark.parametrize("env,wrapper_type,result",
[
    (lambda: PixelObservationWrapper(gym.make("CartPole-v0")), ClipAction, False),
    (lambda: PixelObservationWrapper(gym.make("CartPole-v0")), PixelObservationWrapper, True),
    (lambda: PixelObservationWrapper(gym.make("CartPole-v0")), PixelObservationWrapper, True),
    # param_requires_atari_py(AtariPreprocessing(gym.make("Breakout-v0")), ClipAction, True),
])
def test_has_wrapper(env, wrapper_type, result):
    assert has_wrapper(env(), wrapper_type) == result
