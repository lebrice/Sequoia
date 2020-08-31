from dataclasses import dataclass
from typing import ClassVar, Dict, List

import gym
from torch import Tensor
from torch.utils.data import DataLoader

from common.transforms import Compose, Transforms
from settings.active.setting import ActiveSetting
from settings.base.environment import ActionType, ObservationType, RewardType
from simple_parsing import choice, list_field
from utils.logging_utils import get_logger

from ..active_dataloader import ActiveDataLoader
from .gym_dataloader import GymDataLoader
from .gym_dataset import GymDataset

logger = get_logger(__file__)

@dataclass
class RLSetting(ActiveSetting[Tensor, Tensor, Tensor]):
    """
    """
    # Class variable holding all the available datasets.
    available_datasets: ClassVar[Dict[str, str]] = {
        "cartpole": "CartPole-v0",
    }
    dataset: str = choice(available_datasets, default="CartPole-v0", alias="train_dataset")


    # Transformations to use. See the Transforms enum for the available values.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels)
    
    def __post_init__(self):
        """Creates a new RL environment / setup. """
        super().__post_init__()

        # Calling this here so we know the dims.
        self.prepare_data()

        self._train_loader: GymDataLoader
        self._val_loader: GymDataLoader
        self._test_loader: GymDataLoader

        train_loader = self.train_dataloader()
        print(train_loader.observation_space)
        print(train_loader.action_space)
        # Starting with batch size fixed to 2 for now.

    @property
    def gym_env_name(self) -> str:
        for formatted_env_name in self.available_datasets.values():
            if self.dataset == formatted_env_name:
                return self.dataset
        return self.available_datasets[self.dataset]

    def prepare_data(self, *args, **kwargs):
        # TODO: What should we be doing here for Gym environments?
        pass
        # self.train_env = GymDataLoader(self.gym_env_name, batch_size=2)
        # self.val_env = GymDataLoader(self.gym_env_name, batch_size=2)
        # self.test_env = GymDataLoader(self.gym_env_name, batch_size=2)

    def train_dataloader(self, *args, **kwargs) -> GymDataLoader:
        self._train_loader = GymDataLoader(
            env=self.gym_env_name,
            batch_size=2,
            num_workers=None,
        )
        return self._train_loader

    def val_dataloader(self, *args, **kwargs) -> GymDataLoader:
        self._val_loader = GymDataLoader(
            env=self.gym_env_name,
            batch_size=2,
            num_workers=None,
        )
        return self._val_loader
    
    def test_dataloader(self, *args, **kwargs) -> GymDataLoader:
        self._test_loader = GymDataLoader(
            env=self.gym_env_name,
            batch_size=2,
            num_workers=None,
        )
        return self._test_loader
