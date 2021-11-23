import d3rlpy
from sklearn.model_selection import train_test_split
from typing import ClassVar, List, Tuple, Dict

from sequoia import Results
from sequoia.settings.base import Setting
from torch.utils.data import DataLoader
from sequoia.settings.base import Method
from dataclasses import dataclass

from simple_parsing.helpers import choice


# TODO: Can't inherit from Results here, not sure why
@dataclass
class OfflineRLResults:
    # Metrics from online testing
    test_rewards: list
    test_episode_length: list
    test_episode_count: list

    # Metrics from offline training: not required
    # [ ...(episode index, {..., metric_name: metric_value, ....}) ...
    offline_metrics: List[Tuple[int, Dict[str, float]]] = None

    objective_name: ClassVar[str] = "Average Reward"

    @property
    def objective(self):
        return sum(self.test_rewards) / len(self.test_rewards)


offline_datasets_from_d3rlpy = {'cartpole-replay', 'cartpole-random'}
other_datasets = {}


@dataclass
class OfflineRLSetting(Setting):
    available_datasets: ClassVar[list] = list(offline_datasets_from_d3rlpy) + list(other_datasets)
    dataset: str = choice(available_datasets, default="cartpole-replay")
    val_size: int = 0.2

    # Only d3rlpy uses these params and they're the only ones
    create_mask: bool = False
    mask_size: int = 1

    def __post_init__(self):
        # Load d3rlpy offline dataset
        if self.dataset in offline_datasets_from_d3rlpy:
            mdp_dataset, self.env = d3rlpy.datasets.get_dataset(self.dataset, self.create_mask, self.mask_size)
            self.train_dataset, self.valid_dataset = train_test_split(mdp_dataset, test_size=self.val_size)

        # Load other dataset types here
        elif self.dataset in other_datasets:
            self.env = None
            self.train_dataset = None
            self.valid_dataset = None

    def train_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=batch_size)

    def apply(self, method: Method["OfflineRLSetting"]) -> OfflineRLResults:
        method.configure(self)

        offline_metrics = method.fit(train_env=self.train_dataset, valid_env=self.valid_dataset)

        # Test
        test_rewards, test_episode_length, test_episode_count = method.test(self.env)
        return OfflineRLResults(test_rewards=test_rewards,
                                test_episode_length=test_episode_length,
                                test_episode_count=test_episode_count,
                                offline_metrics=offline_metrics)
