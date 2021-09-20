import d3rlpy
from d3rlpy import algos
from dataclasses import dataclass
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


@dataclass
class OfflineRLSetting(Setting):
    available_datasets: ClassVar[List[str]] = ["hopper-medium-v0"]
    dataset: str = choice(available_datasets)
    test_steps: int = 10_000
    seed: int = 123

    def __post_init__(self):
        torch.random.manual_seed(self.seed)

    def prepare_data(self):
        self.dataset, self.env = d3rlpy.datasets.get_dataset(self.dataset)

    def setup(self):
        self.train_dataset, self.valid_dataset = random_split(self.dataset) 
        pass

    def train_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=batch_size)

    def apply(self, method: Method["OfflineRLSetting"]) -> Results:
        method.configure(self)
        method.fit(train_env=self.train_dataloader(), valid_env=self.valid_dataloader())

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


class SAC(Method, target_setting=OfflineRLSetting):
    def __init__(self, train_steps: int = 1_000_000):
        super().__init__()
        self.train_steps = train_steps
        self.algo: SAC

    def configure(self, setting: OfflineRLSetting):
        super().configure(setting)
        # prepare algorithm
        self.algo = SAC()

    def fit(self, train_env: MDPDataset, valid_env: MDPDataset) -> None:
        # train offline
        self.algo.fit(dataset, n_steps=self.train_steps)
        # self.algo.evaluate(valid_env)
        # train online
        sac.fit_online(env, n_steps=1000000)
    
    def get_actions(self, obs: np.ndarray, action_space: Space) -> np.ndarray:
        # ready to control
        return self.algo.predict(obs)


def main():
    setting = OfflineRLSetting(dataset="hopper-medium-v0")
    method = SAC()
    results = setting.apply(method)
    print(results)

if __name__ == "__main__":
    main()
