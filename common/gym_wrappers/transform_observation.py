from gym.wrappers import TransformObservation as TransformObservation_
import gym
from common.transforms import Compose
from typing import List, Callable, Union
from utils.logging_utils import get_logger

logger = get_logger(__file__)

class TransformObservation(TransformObservation_):
    def __init__(self, env: gym.Env, f: Union[Callable, List[Callable]]):
        if isinstance(f, list) and not callable(f):
            f = Compose(f)
        super().__init__(env, f=f)
        self.observation_space = self.env.observation_space
        self.f: Compose
        self.observation_space.shape = self.f.shape_change(self.observation_space.shape)
        logger.debug(f"New observation shape after transforms: {self.observation_space}")



