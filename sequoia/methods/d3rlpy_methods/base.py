from typing import Type, ClassVar

import gym
from d3rlpy.algos import *
import numpy as np
from gym import Space

from sequoia import Method
from sequoia.settings.offline_rl.setting import OfflineRLSetting


class OfflineRLWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.x

        # TODO: If action space is changed to dictionary, do this
        # self.action_space = env.action_space.y_pred

    def reset(self):
        observation = super().reset()
        return observation.x

    def step(self, action):
        # TODO: if step expects a dictionary as action, just pass {'y_pred': action}
        observation, reward, done, info = super().step(action)
        return observation.x, reward.y, done, info


class BaseOfflineRLMethod(Method, target_setting=OfflineRLSetting):
    Algo: ClassVar[Type[AlgoBase]] = AlgoBase

    def __init__(self, train_steps: int = 1_000_000, n_epochs: int = 5, scorers: dict = None):
        super().__init__()
        self.train_steps = train_steps
        self.n_epochs = n_epochs
        self.scorers = scorers
        self.algo = type(self).Algo()

    def configure(self, setting: OfflineRLSetting):
        super().configure(setting)
        self.setting = setting

    def fit(self, train_env, valid_env) -> None:
        if isinstance(self.setting, OfflineRLSetting):
            self.algo.fit(train_env,
                          eval_episodes=valid_env,
                          n_epochs=self.n_epochs,
                          scorers=self.scorers)
        else:
            train_env, valid_env = OfflineRLWrapper(train_env), OfflineRLWrapper(valid_env)
            self.algo.fit_online(env=train_env, eval_env=valid_env, n_steps=self.train_steps)

    def get_actions(self, obs: np.ndarray, action_space: Space) -> np.ndarray:
        # ready to control
        return self.algo.predict(obs)



"""
D3RLPY Methods: work on OfflineRL and TraditionalRL assumptions
"""


class DQNMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DQN


class DoubleDQNMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DoubleDQN


class DDPGMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DDPG


class TD3Method(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = TD3


class SACMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = SAC


class DiscreteSACMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DiscreteSAC


class CQLMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = CQL


class DiscreteCQLMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DiscreteCQL


class BEAR(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = BEAR


class AWRMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = AWR


class DiscreteAWRMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DiscreteAWR


class BC(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = BC


class DiscreteBCMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DiscreteBC


class BCQMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = BCQ


class DiscreteBCQMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DiscreteBCQ
