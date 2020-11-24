""" Fixes some of the annoying things about the PixelObservationWrapper. """
from typing import Union

import gym
import numpy as np
from common.transforms import to_tensor
from gym.envs.classic_control import CartPoleEnv
from gym.wrappers.pixel_observation import PixelObservationWrapper as PixelObservationWrapper_
from torch import Tensor


class PixelObservationWrapper(PixelObservationWrapper_):
    """ Less annoying version of gym's `PixelObservationWrapper`:

    - Resets the environment before calling the constructor (fixes crash).
    - Makes the popup window non-visible when rendering with mode="rgb_array".
    - State is always pixels instead of dict with pixels at key 'pixels'
        - TODO: What if we wanted to also have access to the state? We might
          have to revert this change at some point.
    - `reset()` returns the pixels.
    """
    def __init__(self, env: Union[str, gym.Env]):
        if isinstance(env, str):
            env = gym.make(env)
        env.reset()
        super().__init__(env)
        self.observation_space = self.observation_space["pixels"]
        from gym.envs.classic_control.rendering import Viewer
        self.viewer: Viewer
        if self.env.viewer is None:
            self.env.render(mode="rgb_array")
                    
        if self.env.viewer is not None:
            self.viewer: Viewer = env.viewer
            self.viewer.window.set_visible(False)
    
    def step(self, *args, **kwargs):
        state, reward, done, info = super().step(*args, **kwargs)
        state = state["pixels"]
        state = self.to_array(state)
        return state, reward, done, info

    def reset(self, *args, **kwargs):
        self.state = super().reset()["pixels"]
        self.state = self.to_array(self.state)
        return self.state
    
    def render(self, mode: str="human", **kwargs):
        if mode == "human" and self.viewer and not self.viewer.window.visible:
            self.viewer.window.set_visible(True)
        return super().render(mode=mode, **kwargs)

    def to_array(self, image) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            # TODO: There is something weird happening here, something to do
            # with the image having a negative stride dimension or something
            # like that. Also, ideally, we would return a numpy array (without
            # depending on pytorch here)
            return to_tensor(image)
            return np.array(image.copy())
        return image
