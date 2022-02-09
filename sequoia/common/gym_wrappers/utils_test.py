import gym
import pytest
from gym.wrappers import ClipAction
from gym.wrappers.pixel_observation import PixelObservationWrapper

from .pixel_observation import PixelObservationWrapper
from .utils import has_wrapper
from sequoia.conftest import param_requires_pyglet


@pytest.mark.parametrize(
    "env,wrapper_type,result",
    [
        param_requires_pyglet(
            lambda: PixelObservationWrapper(gym.make("CartPole-v0")), ClipAction, False
        ),
        param_requires_pyglet(
            lambda: PixelObservationWrapper(gym.make("CartPole-v0")), PixelObservationWrapper, True
        ),
        param_requires_pyglet(
            lambda: PixelObservationWrapper(gym.make("CartPole-v0")), PixelObservationWrapper, True
        ),
        # param_requires_atari_py(AtariPreprocessing(gym.make("ALE/Breakout-v5")), ClipAction, True),
    ],
)
def test_has_wrapper(env, wrapper_type, result):
    assert has_wrapper(env(), wrapper_type) == result
