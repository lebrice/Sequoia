import gym
import numpy as np
from sequoia.common.spaces import Image
from sequoia.common.transforms import Compose, Transforms
from sequoia.conftest import monsterkong_required

from .transform_wrappers import TransformObservation


@monsterkong_required
def test_compose_on_image_space():
    in_space = Image(0, 255, shape=(64, 64, 3), dtype=np.uint8)
    transform = Compose([Transforms.to_tensor, Transforms.three_channels])
    expected = Image(0, 1., shape=(3, 64, 64), dtype=np.float32) 
    actual = transform(in_space)
   
    assert actual == expected
    env = gym.make("MetaMonsterKong-v0")
    assert env.observation_space == gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
    assert env.observation_space == in_space
    wrapped_env = TransformObservation(env, transform)
    assert wrapped_env.observation_space == expected
