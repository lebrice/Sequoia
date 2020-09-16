import gym
from gym.wrappers import FilterObservation, ClipAction, AtariPreprocessing
from gym.wrappers.pixel_observation import PixelObservationWrapper
import pytest

from .pixel_state import PixelStateWrapper

from .utils import wrapper_is_present

@pytest.mark.parametrize("env,wrapper_type,result",
[
    (PixelStateWrapper(gym.make("CartPole-v0")), ClipAction, False),
    (PixelStateWrapper(gym.make("CartPole-v0")), PixelStateWrapper, True),
    (PixelStateWrapper(gym.make("CartPole-v0")), PixelObservationWrapper, True),
    # (AtariPreprocessing(gym.make("Breakout-v0")), ClipAction, True),
])
def test_wrapper_is_present(env, wrapper_type, result):
    assert wrapper_is_present(env, wrapper_type) == result
