from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple

import gym
from torch import Tensor
from torch.utils.data import DataLoader

from common.config import Config
from common.transforms import Compose, Transforms
from settings.active.setting import ActiveSetting
from settings.base.environment import ActionType, ObservationType, RewardType
from simple_parsing import choice, list_field
from utils import Parseable, Serializable, dict_union
from utils.json_utils import Pickleable
from utils.logging_utils import get_logger

from ..active_dataloader import ActiveDataLoader
from .gym_dataloader import GymDataLoader
from .gym_dataset import GymDataset

logger = get_logger(__file__)

@dataclass
class SettingOptions(Serializable, Parseable, Pickleable):
    pass

@dataclass
class RLSetting(ActiveSetting[Tensor, Tensor, Tensor], Pickleable):
    """
    """
    # Class variable holding all the available datasets.
    available_datasets: ClassVar[Dict[str, str]] = {
        "cartpole": "CartPole-v0",
    }
    observe_state_directly: bool = False
    dataset: str = choice(available_datasets, default="CartPole-v0")

    # Transformations to use. See the Transforms enum for the available values.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels)
    
    # Starting with batch size fixed to 2 for now.
    batch_size: int = 2

    def __post_init__(self):
        """Creates a new RL environment / setup. """
        # FIXME: Do we really need to pass the dims to the parent constructor?
        # I don't think so, so we should get rid of that.
        # This is a bit annoying to have to do, but whatever.
        train_loader = self.train_dataloader()
        logger.debug(f"observation_space: {train_loader.observation_space}")
        logger.debug(f"action_space: {train_loader.action_space}")

        obs_shape: Tuple[int, ...] = train_loader.observation_space.shape
        action_shape: Tuple[int, ...] = train_loader.action_space.shape
        # NOTE: We assume scalar rewards for now.
        reward_shape: Tuple[int, ...] = (1,)
        
        super().__post_init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
        )

        logger.debug(f"self.obs_shape: {self.obs_shape}")
        logger.debug(f"self.action_shape: {self.action_shape}")
        logger.debug(f"self.reward_shape: {self.reward_shape}")

        self.dataloader_kwargs = dict(
            batch_size=self.batch_size,
        )
        self.dims = self.obs_shape

        logger.debug("__post_init__ of RL setting")
        self._train_loader: GymDataLoader
        self._val_loader: GymDataLoader
        self._test_loader: GymDataLoader

        

    def configure(self, config: Config, **dataloader_kwargs):
        """ Set some of the misc options in the setting which might come from
        the Method or the Experiment.
        
        TODO: This isn't super clean, but we basically want to be able to give
        the batch_size, data_dir, num_workers etc to the Setting somehow,
        without letting it "know" what kind of method is being applied to it.
        """
        self.config = config
        self.data_dir = config.data_dir
        # self.dataloader_kwargs.update(num_workers=config.num_workers)
        self.dataloader_kwargs.update(dataloader_kwargs)
        self.batch_size = self.dataloader_kwargs.get("batch_size", self.batch_size)
        self._configured = True

    @property
    def gym_env_name(self) -> str:
        for formatted_env_name in self.available_datasets.values():
            if self.dataset == formatted_env_name:
                return self.dataset
        return self.available_datasets[self.dataset]

    def setup(self, stage=None):
        # TODO: What should we be doing here for Gym environments?
        return super().setup(stage=stage)

    def prepare_data(self, *args, **kwargs):
        # TODO: What should we be doing here for Gym environments?
        super().prepare_data(*args, **kwargs)

    def train_dataloader(self, *args, **kwargs) -> GymDataLoader:
        if args or kwargs:
            logger.warning(UserWarning(
                f"Ignoring args {args} and kwargs {kwargs} for now."
            ))
        self._train_loader = GymDataLoader(
            env=self.gym_env_name,
            batch_size=self.batch_size,
            num_workers=self.batch_size,
            observe_pixels=not self.observe_state_directly,
        )
        return self._train_loader

    def val_dataloader(self, *args, **kwargs) -> GymDataLoader:
        if args or kwargs:
            logger.warning(UserWarning(
                f"Ignoring args {args} and kwargs {kwargs} for now."
            ))
        self._val_loader = GymDataLoader(
            env=self.gym_env_name,
            batch_size=self.batch_size,
            num_workers=self.batch_size,
            observe_pixels=not self.observe_state_directly,
        )
        return self._val_loader

    def test_dataloader(self, *args, **kwargs) -> GymDataLoader:
        if args or kwargs:
            logger.warning(UserWarning(
                f"Ignoring args {args} and kwargs {kwargs} for now."
            ))
        self._test_loader = GymDataLoader(
            env=self.gym_env_name,
            batch_size=self.batch_size,
            num_workers=self.batch_size,
            observe_pixels=not self.observe_state_directly,
        )
        return self._test_loader

    @property
    def train_env(self) -> GymDataLoader:
        return self._train_loader

    @property
    def val_env(self) -> GymDataLoader:
        return self._train_loader

    @property
    def test_env(self) -> GymDataLoader:
        return self._train_loader


    def train_send(self, action: Tensor) -> Tensor:
        """ Send a batch of actions to the train environment/ Active DataLoader.

        Returns the batch of rewards.
        """
        assert self._train_loader is not None
        return self._train_loader.send(action)

    def val_send(self, action: Tensor) -> Tensor:
        """ Send a batch of actions to the val environment/ Active DataLoader.

        Returns the batch of rewards.
        """
        assert self._val_loader is not None
        return self._val_loader.send(action)
    
    def test_send(self, action: Tensor) -> Tensor:
        """ Send a batch of actions to the test environment/ Active DataLoader.

        Returns the batch of rewards.
        """
        assert self._test_loader is not None
        return self._test_loader.send(action)
