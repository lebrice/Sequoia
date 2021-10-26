import d3rlpy
import torch
from d3rlpy import algos
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from typing import ClassVar, List, Type

from sequoia import TraditionalRLSetting, RLEnvironment
from sequoia.settings.base import Setting, Results
from d3rlpy.algos import SAC, DQN, CQL, DiscreteCQL, BC, DiscreteSAC, AlgoBase
from simple_parsing.helpers import choice
from torch.utils.data import DataLoader
from sequoia.settings.base import Method
import numpy as np
from gym.spaces import Space
import gym
from dataclasses import dataclass
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sequoia.common.gym_wrappers.transform_wrappers import TransformObservation, TransformReward
from sequoia.settings.rl.wrappers.measure_performance import MeasureRLPerformanceWrapper
import gym
from simple_parsing.helpers import choice

@dataclass
class OfflineRLSetting(Setting):
    available_datasets: ClassVar[dict] = {"CartPole-v0": d3rlpy.datasets.get_cartpole}
    dataset: str = choice(available_datasets.keys(), default="CartPole-v0")
    val_size: int = 0.2
    test_steps: int = 10_000
    seed: int = 123

    def train_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=batch_size)

    def apply(self, method: Method["OfflineRLSetting"]) -> Results:
        method.configure(self)
        self.dataset, self.env = self.available_datasets[self.dataset]()
        self.train_dataset, self.valid_dataset = train_test_split(self.dataset, test_size=self.val_size)
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


class SACMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = SAC


class DiscreteSACMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = DiscreteSAC


class CQLMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = CQL


class DiscreteCQLMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = DiscreteCQL


class BehaviorCloningMethod(BaseOfflineRLMethod):
    Algo: Type[AlgoBase] = BC


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
