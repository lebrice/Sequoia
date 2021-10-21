import d3rlpy
import torch
from d3rlpy import algos
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from typing import ClassVar, List

from sequoia import TraditionalRLSetting
from sequoia.settings.base import Setting, Results
from d3rlpy.dataset import MDPDataset
from torch.utils.data import random_split
from gym.wrappers import Monitor
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from d3rlpy.algos import SAC, DQN
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


@dataclass
class OfflineRLSetting(Setting):
    # available_datasets: ClassVar[List[str]] = ["CartPole-v0"]
    # dataset: str = choice(available_datasets)
    dataset: str = "CartPole-v0"
    val_size: int = 0.2
    test_steps: int = 10_000
    seed: int = 123

    def train_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=batch_size)

    def apply(self, method: Method["OfflineRLSetting"]) -> Results:
        method.configure(self)
        if self.dataset == "CartPole-v0":
            self.dataset, self.env = d3rlpy.datasets.get_cartpole()
        else:
            self.dataset, self.env = d3rlpy.datasets.get_dataset(self.dataset)

        self.train_dataset, self.valid_dataset = train_test_split(self.dataset, test_size=self.val_size)
        method.fit(train_env=self.train_dataset,
                   valid_env=self.valid_dataset)


class BaseOfflineRLMethod(Method, target_setting=OfflineRLSetting):
    def __init__(self, train_steps: int = 1_000_000, n_epochs: int = 5, scorers: dict = None):
        super().__init__()
        self.train_steps = train_steps
        self.n_epochs = n_epochs
        self.scorers = scorers

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
            #
            # Wrap train_env and valid_env
            # we have: <class 'sequoia.settings.rl.wrappers.measure_performance.MeasureRLPerformanceWrapper'>
            # we require: <class 'gym.wrappers.time_limit.TimeLimit'>

            self.algo.fit_online(env=train_env, eval_env=valid_env, n_steps=self.train_steps)

    def get_actions(self, obs: np.ndarray, action_space: Space) -> np.ndarray:
        # ready to control
        return self.algo.predict(obs)


class DQNMethod(BaseOfflineRLMethod):
    def __init__(self, train_steps: int = 1_000_000, n_epochs: int = 5, scorers: dict = None):
        super().__init__(train_steps, n_epochs, scorers)
        self.algo = DQN()


def main():
    setting_offline = OfflineRLSetting(dataset="CartPole-v0")
    setting_online = TraditionalRLSetting(dataset="CartPole-v0")
    method = DQNMethod(scorers={
        'td_error': td_error_scorer,
        'value_scale': average_value_estimation_scorer
    })
    # _ = setting_offline.apply(method)

    # Not working
    results = setting_online.apply(method)
    # print(results)


if __name__ == "__main__":
    main()
