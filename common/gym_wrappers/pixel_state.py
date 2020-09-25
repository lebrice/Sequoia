""" Fixes some of the annoying things about the PixelObservationWrapper. """
from typing import Union

import gym
import numpy as np
from common.transforms import to_tensor
from gym.envs.classic_control import CartPoleEnv
from gym.envs.classic_control.rendering import Viewer
from gym.wrappers.pixel_observation import PixelObservationWrapper
from torch import Tensor


class PixelStateWrapper(PixelObservationWrapper):
    """ Less annoying version of `PixelObservationWrapper`:

    - Resets the environment before calling the constructor.
    - Makes the popup window non-visible when rendering with mode="rgb_array".
    - State is always pixels instead of dict with pixels at key 'pixels'
    - `reset()` returns the pixels.
    """
    def __init__(self, env: Union[str, gym.Env]):
        if isinstance(env, str):
            env = gym.make(env)
        env.reset()
        super().__init__(env)
        self.observation_space = self.observation_space["pixels"]
        self.viewer: Viewer
        if self.env.viewer is None:
            self.env.render(mode="rgb_array")
        if self.env.viewer is not None:
            self.viewer: Viewer = env.viewer
            self.viewer.window.set_visible(False)
    
    def step(self, *args, **kwargs):
        state, reward, done, info = super().step(*args, **kwargs)
        state = state["pixels"]
        state = self.to_tensor(state)
        return state, reward, done, info

    def reset(self, *args, **kwargs):
        self.state = super().reset()["pixels"]
        self.state = self.to_tensor(self.state)
        return self.state
    
    def render(self, *args, mode: str="human", **kwargs):
        if mode == "human" and self.viewer and not self.viewer.window.visible:
            self.viewer.window.set_visible(True)
        return super().render(*args, mode=mode, **kwargs)

    def to_tensor(self, image: np.ndarray):
        if not isinstance(image, Tensor):
            return to_tensor(image)
        return image
