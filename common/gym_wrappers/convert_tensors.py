from functools import singledispatch, wraps
from typing import Any, Dict, List, Tuple, TypeVar, Union

import gym
import numpy as np
import torch
from gym import Space, spaces
from torch import Tensor
from utils.move import move
from utils.logging_utils import get_logger

logger = get_logger(__file__)

S = TypeVar("S", bound=Space)


class ConvertToFromTensors(gym.Wrapper):
    """ Wrapper that converts Tensors into samples/ndarrays and vice versa.
    
    Whatever comes into the env is converted into np.ndarrays or samples from
    the action space, and whatever comes out of the environment (observations,
    rewards, dones, etc.) get converted to Tensors.
    
    Also supports Dict/Tuple/etc observation/action spaces.
    
    Also makes it so the `sample` methods of both the observation and
    action spaces return Tensors, and that their `contains` methods also accept
    Tensors as an input.
    
    If `device` is given, created Tensors are moved to the provided device.
    """
    def __init__(self, env: gym.Env, device: Union[torch.device, str] = None):
        super().__init__(env=env)
        self.device = device
        self.observation_space: Space = wrap_space(self.env.observation_space, device=device)
        self.action_space: Space = wrap_space(self.env.action_space, device=device)
        if hasattr(self.env, "reward_space"):
            self.reward_space: Space = wrap_space(self.env.reward_space, device=device)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return to_tensor(self.observation_space, obs, device=self.device)

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Dict]]:
        action = from_tensor(self.action_space, action)

        result = self.env.step(action)
        observation, reward, done, info = result
        
        observation = to_tensor(self.observation_space, observation, self.device)

        if hasattr(self, "reward_space"):
            reward = to_tensor(self.reward_space, reward, self.device)
        else:
            reward = torch.as_tensor(reward, device=self.device)
        done = torch.as_tensor(done, device=self.device)
        # We could actually do this!
        # info = np.ndarray(info)
        return type(result)([observation, reward, done, info])


def wrap_space(space: S, device: torch.device = None) -> S:
    """Wraps `space` so its `sample()` method produces Tensors, and its
    `contains` method also accepts Tensors.
    
    Returns the modified Space.
    """
    # Save the original methods so we can use them.
    sample = space.sample
    contains = space.contains
    
    @wraps(space.sample)
    def _sample(*args, **kwargs):
        samples = sample(*args, **kwargs)
        samples = to_tensor(space, samples)
        if device:
            samples = move(samples, device)
        return samples

    @wraps(space.contains)
    def _contains(x: Union[Tensor, Any]) -> bool:
        x = from_tensor(space, x)
        return contains(x)

    space.sample = _sample
    space.contains = _contains

    return space

@singledispatch
def from_tensor(space: Space, sample: Union[Tensor, Any]) -> Union[np.ndarray, Any]:
    """ Converts a Tensor into a sample from the given space. """
    if isinstance(sample, Tensor):
        return sample.cpu().numpy()
    return sample


@from_tensor.register
def _(space: spaces.Discrete, sample: Tensor) -> int:
    if isinstance(sample, Tensor):
        return sample.item()
    return sample


@from_tensor.register
def _(space: spaces.Dict, sample: Dict[str, Union[Tensor, Any]]) -> Dict[str, Union[np.ndarray, Any]]:
    return {
        key: from_tensor(space[key], value)
        for key, value in sample.items()
    }

@from_tensor.register
def _(space: spaces.Tuple, sample: Tuple[Union[Tensor, Any]]) -> Tuple[Union[np.ndarray, Any]]:
    return type(sample)(
        from_tensor(space[i], value)
        for i, value in enumerate(sample)
    )


@singledispatch
def to_tensor(space: Space,
              sample: Union[np.ndarray, Any],
              device: torch.device = None) -> Union[np.ndarray, Any]:
    """ Converts a sample from the given space into a Tensor. """
    return torch.as_tensor(sample, device=device)


@to_tensor.register
def _(space: spaces.MultiBinary,
      sample: np.ndarray,
      device: torch.device = None) -> Dict[str, Union[Tensor, Any]]:
    return torch.as_tensor(sample, device=device, dtype=torch.bool)


@to_tensor.register
def _(space: spaces.Dict,
      sample: Dict[str, Union[np.ndarray, Any]],
      device: torch.device = None) -> Dict[str, Union[Tensor, Any]]:
    return {
        key: to_tensor(space[key], value, device)
        for key, value in sample.items()
    }


@to_tensor.register
def _(space: spaces.Tuple,
      sample: Tuple[Union[np.ndarray, Any], ...],
      device: torch.device = None) -> Tuple[Union[Tensor, Any], ...]:
    return type(sample)(
        to_tensor(space[i], value, device)
        for i, value in enumerate(sample)
    )
