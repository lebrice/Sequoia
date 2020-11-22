from .convert_tensors import ConvertToFromTensors, wrap_space, to_tensor, from_tensor
import gym
from gym import spaces
import torch
from torch import Tensor
import numpy as np
from typing import Union



def test_wrap_space():
    space = spaces.Box(0, 1, (28,28), dtype=np.uint8)
    new_space = wrap_space(space)
    sample = new_space.sample()

    assert isinstance(sample, Tensor)
    assert sample in new_space
    assert sample in new_space
    assert sample.dtype == torch.uint8

import pytest


@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
def test_convert_tensors_wrapper(device: Union[str, torch.device]):
    if device == "cuda": #and not torch.cuda.is_available():
        pytest.skip("Cuda isn't available on this machine.")
        return # unneeded, but just to illustrate that this exits the test.

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
