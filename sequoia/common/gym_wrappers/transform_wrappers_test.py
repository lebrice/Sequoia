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
    expected = Image(0, 1.0, shape=(3, 64, 64), dtype=np.float32)
    actual = transform(in_space)

    assert actual == expected
    env = gym.make("MetaMonsterKong-v0")
    assert env.observation_space == gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
    assert env.observation_space == in_space
    wrapped_env = TransformObservation(env, transform)
    assert wrapped_env.observation_space == expected


import pytest
import torch
from torchvision.datasets import MNIST

from sequoia.common.transforms import Compose


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need cuda for this test.")
def test_move_wrapper_and_iteration():
    batch_size = 1
    transforms = Compose([Transforms.to_tensor])
    dataset = MNIST("data", transform=transforms)
    obs_space = Image(0, 255, (1, 28, 28), np.uint8)
    obs_space = transforms(obs_space)
    from sequoia.settings.sl.environment import PassiveEnvironment

    env = PassiveEnvironment(
        dataset,
        batch_size=batch_size,
        n_classes=10,
        observation_space=obs_space,
    )

    from functools import partial

    from sequoia.utils.generic_functions import move

    from .transform_wrappers import TransformReward

    env = TransformObservation(env, partial(move, device="cuda"))
    env = TransformReward(env, partial(move, device="cuda"))

    obs, rewards_next = next(iter(env))
    rewards_send = env.send(env.action_space.sample())
    assert obs.device.type == "cuda"
    assert rewards_next.device.type == "cuda"
    assert rewards_send.device.type == "cuda"
