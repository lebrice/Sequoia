import d3rlpy
from sklearn.model_selection import train_test_split
from typing import ClassVar, List, Tuple, Dict


from sequoia.settings.base import Setting
from torch.utils.data import DataLoader
from sequoia.settings.base import Method
from dataclasses import dataclass

from simple_parsing.helpers import choice


@dataclass
class OfflineRLSetting(Setting):
    available_datasets: ClassVar[list] = ["cartpole-replay",  # Cartpole Replay
                                          "cartpole-random",  # Cartpole Random
                                          ]
    dataset: str = choice(available_datasets, default="cartpole-replay")
    create_mask: bool = False
    mask_size: int = 1
    val_size: int = 0.2
    seed: int = 123

    def __post_init__(self):
        self.mdp_dataset, self.env = d3rlpy.datasets.get_dataset(self.dataset, self.create_mask, self.mask_size)
        self.train_dataset, self.valid_dataset = train_test_split(self.mdp_dataset, test_size=self.val_size)

    def train_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=batch_size)

    def apply(self, method: Method["OfflineRLSetting"]) -> List[Tuple[int, Dict[str, float]]]:
        method.configure(self)
        return method.fit(train_env=self.train_dataset, valid_env=self.valid_dataset)



