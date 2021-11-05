import d3rlpy
from sklearn.model_selection import train_test_split
from typing import ClassVar, List, Type

from sequoia import TraditionalRLSetting
from sequoia.methods.d3rlpy_methods.base import SACMethod
from sequoia.settings.base import Setting, Results
from torch.utils.data import DataLoader
from sequoia.settings.base import Method
from dataclasses import dataclass
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from simple_parsing.helpers import choice

@dataclass()
class OfflineRLResults(Results):
    objective_name: ClassVar[str] = "Average Reward"
    rewards: List[int]

    @property
    def objective(self):
        assert self.rewards
        return sum(self.rewards)/len(self.rewards)





@dataclass
class OfflineRLSetting(Setting):
    # We can pass in any of the available datasets shown below.
    # get_dataset() will attempt to match the regex expressions for d4rl, py_bullet and atari

    """ TODO: consult with fabrice about regular expressions the best way to show users how to use
        https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/datasets.py
     """

    # TODO: Results: ClassVar[Type[Results]] = OfflineRLResults

    available_datasets: ClassVar[list] = ["cartpole-replay",  # Cartpole Replay
                                          "cartpole-random",  # Cartpole Random
                                          "pendulum-replay",  # Pendulum Replay
                                          "pendulum-random",  # Pendulum Random
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


"""
Quick example using DQN for offline cart-pole

def main():
    setting_offline = OfflineRLSetting(dataset="cartpole-replay")
    setting_online = TraditionalRLSetting(dataset="CartPole-v0")
    method = SACMethod(scorers={
        'td_error': td_error_scorer,
        'value_scale': average_value_estimation_scorer
    })

    _ = setting_offline.apply(method)
    # results = setting_online.apply(method)
    # print(results)


if __name__ == "__main__":
    main()

"""
