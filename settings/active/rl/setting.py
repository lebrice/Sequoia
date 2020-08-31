from dataclasses import dataclass
from typing import ClassVar, Dict, List

import gym
from torch.utils.data import DataLoader

from simple_parsing import choice, list_field

from settings.base.environment import ActionType, ObservationType, RewardType
from settings.active.setting import ActiveSetting
from common.transforms import Compose, Transforms
from 

@dataclass
class RLSetting(ActiveSetting[Tensor, Tensor, Tensor]):
    """
    """

    # Class variable holding all the available datasets.
    available_datasets: ClassVar[Dict[str, str]] = {
        "cartpole": "CartPole-v0",
    }
    dataset: str = choice(available_datasets.keys(), default="cartpole")

    # Transformations to use. See the Transforms enum for the available values.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels)
    
    def __post_init__(self):
        """Creates a new RL environment / setup. """
        super().__post_init__()
        # Starting without batching for now.
        self.train_env: GymEnv
        self.val_env:   GymEnv
        self.test_env:  GymEnv

        # Calling this here so we know the dims.
        self.prepare_data()
        print(self.train_env.observation_space)
        print(self.train_env.action_space)
        exit()

    def prepare_data(self, *args, **kwargs):
        gym_env_name = self.available_datasets[self.dataset]
        self.train_env = GymDataLoader(gym_env_name)
        self.val_env   = GymDataLoader(gym_env_name)
        self.test_env  = GymDataLoader(gym_env_name)

    def train_dataloader(self) -> ActiveEnvironment[ObservationType, ActionType, RewardType]:
        return DataLoader(self.train_env, num_workers=0, batch_size=None)
