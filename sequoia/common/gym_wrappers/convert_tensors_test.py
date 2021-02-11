from .convert_tensors import (
    ConvertToFromTensors,
    add_tensor_support,
    to_tensor,
    from_tensor,
)
import gym
from gym import spaces
import torch
from torch import Tensor
import numpy as np
from typing import Union


def test_add_tensor_support():
    space = spaces.Box(0, 1, (28, 28), dtype=np.uint8)
    new_space = add_tensor_support(space)
    sample = new_space.sample()

    assert isinstance(sample, Tensor)
    assert sample in new_space
    assert sample in new_space
    assert sample.dtype == torch.uint8


import pytest
from sequoia.conftest import skipif_param


@pytest.mark.parametrize(
    "device",
    [
        None,
        "cpu",
        skipif_param(
            not torch.cuda.is_available(),
            "cuda",
            reason="Cuda is required for this test",
        ),
    ],
)
def test_convert_tensors_wrapper(device: Union[str, torch.device]):
    env_name = "Breakout-v0"
    env = gym.make(env_name)
    env = ConvertToFromTensors(env, device=device)
    obs = env.reset()
    assert isinstance(obs, Tensor)
    if device:
        assert obs.device.type == device

    action = env.action_space.sample()
    obs, reward, done, info = env.step(torch.as_tensor(action))
    assert isinstance(obs, Tensor)
    assert isinstance(reward, Tensor)
    assert isinstance(done, Tensor)
    if device:
        assert obs.device.type == device
        assert reward.device.type == device
        assert done.device.type == device
