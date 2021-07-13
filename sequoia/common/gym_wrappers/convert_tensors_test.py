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
    env_name = "CartPole-v0"
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
    # TODO: Not quite sure this is the best thing to do:
    # assert isinstance(done, Tensor) # not sure this is useful!
    if device:
        assert obs.device.type == device
        assert reward.device.type == device
        # assert done.device.type == device


from sequoia.common.spaces import NamedTupleSpace, TypedDictSpace
from sequoia.common.batch import Batch
from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Foo(Batch):
    x: Tensor
    task_labels: Optional[Tensor]


def test_preserves_dtype_of_namedtuple_space():    
    input_space = NamedTupleSpace(
        x=spaces.Box(0, 1, [32, 123, 123, 3]),
        task_labels=spaces.MultiDiscrete([5 for _ in range(32)]),
        dtype=Foo,
    )

    output_space = add_tensor_support(input_space)
    assert output_space.dtype is input_space.dtype


def test_preserves_dtype_of_typeddict_space():    
    input_space = TypedDictSpace(
        x=spaces.Box(0, 1, [32, 123, 123, 3]),
        task_labels=spaces.MultiDiscrete([5 for _ in range(32)]),
        dtype=Foo,
    )
    output_space = add_tensor_support(input_space)
    assert output_space.dtype is input_space.dtype