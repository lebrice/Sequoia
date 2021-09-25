import d3rlpy
import torch
from d3rlpy import algos
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from typing import ClassVar, List
from sequoia.settings.base import Setting, Results
from d3rlpy.dataset import MDPDataset
from torch.utils.data import random_split
from gym.wrappers import Monitor
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from d3rlpy.algos import SAC
from simple_parsing.helpers import choice
from torch.utils.data import DataLoader
from sequoia.settings.base import Method
import numpy as np
from gym.spaces import Space
import gym
from dataclasses import dataclass
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer


@dataclass
class OfflineRLSetting(Setting):
    # available_datasets: ClassVar[List[str]] = ["CartPole-v0"]
    # dataset: str = choice(available_datasets)
    dataset: str = "CartPole-v0"
    val_size: int = 0.1
    test_steps: int = 10_000
    seed: int = 123

    def __post_init__(self):
        torch.random.manual_seed(self.seed)
        self.train_dataset, self.valid_dataset = train_test_split(self.dataset, test_size=self.val_size)

    def prepare_data(self, dataset):
        self.dataset, self.env = d3rlpy.datasets.get_cartpole(self.dataset)

    def train_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=batch_size)

    def apply(self, method: Method["OfflineRLSetting"]) -> Results:
        method.configure(self)

        method.fit(train_env=self.train_dataset, valid_env=self.valid_dataset)

        self.env.seed(self.seed)
        self.env = Monitor(self.env, directory="/results")


        while steps < self.test_steps and self.env.is_closed():
            obs = self.env.reset()
            done = False
            while not done:
                action = method.get_actions(obs, action_space=self.action_space)
                obs, reward, done, info = self.env.step()
        # todo: get the statistics from the env.
        stats = self.env.get_results()
        result = self.Results(stats)
        return result


class SACMethod(Method, target_setting=OfflineRLSetting):
    def __init__(self, train_steps: int = 1_000_000, n_epochs: int = 100, scorers: dict = None):
        super().__init__()
        self.train_steps = train_steps
        self.n_epochs = n_epochs
        self.scorers = scorers
        self.algo: SAC

    def configure(self, setting: OfflineRLSetting):
        super().configure(setting)
        # prepare algorithm
        self.algo = SAC()

    def fit(self, train_env: MDPDataset, valid_env: MDPDataset) -> None:
        # train offline
        self.algo.fit(train_env,
                      eval_episodes=valid_env,
                      n_steps=self.train_steps,
                      n_epochs=self.n_epochs,
                      scorers=self.scorers)

    def get_actions(self, obs: np.ndarray, action_space: Space) -> np.ndarray:
        # ready to control
        return self.algo.predict(obs)


def main():
    setting = OfflineRLSetting(dataset="CartPole-v0")
    method = SACMethod(scorers={
            'td_error': td_error_scorer,
            'value_scale': average_value_estimation_scorer
        })
    results = setting.apply(method)
    print(results)


if __name__ == "__main__":
    main()
