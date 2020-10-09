import gym
import numpy as np
import torch
from torch import Tensor

from utils.logging_utils import get_logger

logger = get_logger(__file__)


class ConvertToFromTensors(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        return (
            observation,
            torch.as_tensor(reward),
            torch.as_tensor(done),
            info,
        )

    def action(self, action):
        if not self.action_space.contains(action):
            original = action
            if isinstance(action, Tensor):
                action = action.cpu().numpy()
            elif isinstance(action, np.ndarray):
                action = action.tolist()
            if not self.action_space.contains(action):
                if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 1:
                    action = action[0]
            if not self.action_space.contains(action):
                assert False, (original, action, self.action_space.contains(action))
        return action
