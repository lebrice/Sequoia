import d3rlpy
import torch
from d3rlpy import algos
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from typing import ClassVar, List, Type

from sequoia import TraditionalRLSetting
from sequoia.settings.base import Setting, Results
from d3rlpy.algos import *
from torch.utils.data import DataLoader
from sequoia.settings.base import Method
import numpy as np
from gym.spaces import Space
from dataclasses import dataclass
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
import gym
from simple_parsing.helpers import choice
import re
# get_cartpole

# get_pendulum


@dataclass
class OfflineRLSetting(Setting):

    # We can pass in any of the available datasets shown below.
    # get_dataset() will attempt to match the regex expressions for d4rl, py_bullet and atari

    """ TODO: consult with fabrice about regular expressions the best way to show users how to use
        https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/datasets.py
     """

    available_datasets: ClassVar[list] = ["cartpole-replay",  # Cartpole Replay
                                          "cartpole-random",  # Cartpole Random
                                          "pendulum-replay",  # Pendulum Replay
                                          "pendulum-random",  # Pendulum Random
                                          r"^bullet-.+$",     # d4rl
                                          r"hopper|halfcheetah|walker|ant",  # also d4rl
                                          r"^.+-bullet-.+$"  # py_bullet
                                          # Atari: see d3rlpy.datasets.ATARI_GAMES
                                          ]
    dataset: str = choice(available_datasets, default="cartpole-replay")
    create_mask: bool = False
    mask_size: int = 1
    val_size: int = 0.2
    test_steps: int = 10_000
    seed: int = 123

    def train_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=batch_size)

    def apply(self, method: Method["OfflineRLSetting"]) -> Results:
        method.configure(self)
        self.mdp_dataset, self.env = d3rlpy.datasets.get_dataset(self.dataset, self.create_mask, self.mask_size)
        self.train_dataset, self.valid_dataset = train_test_split(self.mdp_dataset, test_size=self.val_size)
        method.fit(train_env=self.train_dataset,
                   valid_env=self.valid_dataset)


class SequoiaToGymWrapper(gym.Wrapper):
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

    Algo: Type[AlgoBase] = AlgoBase

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
            train_env, valid_env = SequoiaToGymWrapper(train_env), SequoiaToGymWrapper(valid_env)
            self.algo.fit_online(env=train_env, eval_env=valid_env, n_steps=self.train_steps)

    def get_actions(self, obs: np.ndarray, action_space: Space) -> np.ndarray:
        # ready to control
        return self.algo.predict(obs)


"""
D3RLPY Methods: work on OfflineRL and TraditionalRL assumptions
"""


class DQNMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = DQN


class DoubleDQNMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = DoubleDQN


class DDPGMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = DDPG


class TD3Method(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = TD3


class SACMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = SAC


class DiscreteSACMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = DiscreteSAC


class CQLMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = CQL


class DiscreteCQLMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = DiscreteCQL


class BEAR(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = BEAR


class AWRMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = AWR

class BC(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = BC


class BCQMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = BCQ


"""
Quick example using DQN for offline cart-pole
"""

def main():
    setting_offline = OfflineRLSetting(dataset="CartPole-v0")
    setting_online = TraditionalRLSetting(dataset="CartPole-v0")
    method = SACMethod(scorers={
        'td_error': td_error_scorer,
        'value_scale': average_value_estimation_scorer
    })

    _ = setting_offline.apply(method)
    results = setting_online.apply(method)
    print(results)


if __name__ == "__main__":
    main()
