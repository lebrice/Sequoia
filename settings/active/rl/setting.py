from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Tuple, Type, Union

import gym
import numpy as np
from gym.spaces import Dict as DictSpace
from gym.wrappers.pixel_observation import PixelObservationWrapper
from pytorch_lightning import Trainer
from torch import Tensor
from torch.utils.data import DataLoader

from common.config import Config
from common.loss import Loss
from common.transforms import Compose, Transforms
from settings.active.setting import ActiveSetting
from settings.base import Results
from settings.base.environment import ActionType, ObservationType, RewardType
from simple_parsing import choice, list_field
from utils import Parseable, Serializable, dict_union
from utils.json_utils import Pickleable
from utils.logging_utils import get_logger

from ..active_dataloader import ActiveDataLoader
from .gym_dataloader import GymDataLoader
from .gym_dataset import GymDataset
from .results import RLResults

logger = get_logger(__file__)

@dataclass
class SettingOptions(Serializable, Parseable, Pickleable):
    # Idea, we could move stuff to 'configuration options' objects instead of
    # having them directly on the Setting.. (This is annoying, but might be
    # necessary because the LightningDataModule class has an __init__ and we
    # use dataclasses.)
    pass


@dataclass
class RLSetting(ActiveSetting[Tensor, Tensor, Tensor], Pickleable):
    """
    """
    results_class: ClassVar[Type[Results]] = RLResults
    # Class variable holding all the available datasets.
    available_datasets: ClassVar[Dict[str, str]] = {
        "cartpole": "CartPole-v0",
        "pendulum": "Pendulum-v0",
    }
    observe_state_directly: bool = False
    dataset: str = choice(available_datasets, default="pendulum")

    # Transformations to use. See the Transforms enum for the available values.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels, Transforms.channels_first)
    
    # Starting with batch size fixed to 2 for now.
    # batch_size: int = 2

    def __post_init__(self):
        """Creates a new RL environment / setup. """
        # FIXME: (@lebrice) This is a bit annoying to have to do, but whatever.
        # The idea is that we want to set some 'dims' attributes so that methods
        # can know what the observations / actions / rewards will look like,
        # even before the dataloaders are created. However in order to do that,
        # we need to actually create an environment to get those shapes from.
        # Btw this also assumes that the shapes don't change between train, val
        # and test (which seems very reasonable for now).
        temp_env: gym.Env = self.create_gym_env(self.gym_env_name)
        temp_env.reset()
        logger.debug(f"train_env observation space: {temp_env.observation_space}")
        assert temp_env.observation_space

        self.action_space = temp_env.action_space
        # Extract the observation space from the env.
        if isinstance(temp_env.observation_space, DictSpace):
            self.observation_space = temp_env.observation_space["pixels"]
        else:
            self.observation_space = temp_env.observation_space

        assert self.observation_space.shape
        obs_shape: Tuple[int, ...] = self.observation_space.shape
        action_shape: Tuple[int, ...] = self.action_space.shape
        # NOTE: We assume scalar rewards for now.
        reward_shape: int = 1
        self.reward_range: Tuple[float, float] = temp_env.reward_range

        temp_env.close()
        del temp_env

        super().__post_init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
        )

        logger.debug(f"obs_shape: {self.obs_shape}")
        logger.debug(f"action_shape: {self.action_shape}")
        logger.debug(f"reward_shape: {self.reward_shape}")

        self.dataloader_kwargs = dict(
            # default batch size?
            # batch_size=32,
        )
        self.dims = self.obs_shape

        logger.debug("__post_init__ of RL setting")
        self._train_loader: GymDataLoader
        self._val_loader: GymDataLoader
        self._test_loader: GymDataLoader

        logger.debug(f"observation_space: {self.observation_space}")
        logger.debug(f"action_space: {self.action_space}")

    def evaluate(self, method: "Method") -> RLResults:
        """Tests the method and returns the Results.

        Overwrite this to customize testing for your experimental setting.

        Returns:
            Results: A Results object for this particular setting.
        """
        from methods import Method
        method: Method
        trainer: Trainer = method.trainer
        test_outputs = trainer.test(
            # TODO: Choose either (or None?)
            datamodule=self,
            # test_dataloaders=self.test_dataloader(),
            verbose=True,
        )
        # TODO: Add this 'evaluate' routine to the CL Trainer?
        # assert False, test_outputs 
        if not test_outputs:
            raise RuntimeError(f"Test outputs should have been produced!")
        
        if isinstance(test_outputs, list):
            assert len(test_outputs) == 1
            test_outputs = test_outputs[0]

        test_loss: Loss = test_outputs["loss_object"]
        mean_reward = test_outputs["mean_reward"]
        hparams = method.hparams

        return self.results_class(
            hparams=hparams,
            test_loss=test_loss,
            mean_reward=mean_reward,
        )
    
    @property
    def gym_env_name(self) -> str:
        if not isinstance(self.dataset, str):
            logger.warning(UserWarning(
                f"Expected self.dataset to be a str, but its {self.dataset}! "
                f"Will try to use it anyway for now."
            ))
            return self.dataset

        env_name: Union[str, Any]
        if self.dataset in self.available_datasets:
            env_name = self.available_datasets[self.dataset]
        elif self.dataset in self.available_datasets.values():
            env_name = self.dataset
        else:
            env_name = self.dataset
            logger.error(
                f"Env {self.dataset} isn't in the list of supported "
                f"environments! (Will try to use it anyway for now)."
            )
        return env_name

    def create_gym_env(self, env_name: str) -> str:
        """ Get the 'formatted' gym environment for `self.dataset`, if needed.
        """
        env = gym.make(env_name)
        env.reset()
        logger.debug(f"spec: {env.spec}, Observation space: {env.observation_space}, action space: {env.action_space}")
        if not self.observe_state_directly:
            # BUG: There is a bug here, the env keeps rendering a screen!
            env = PixelObservationWrapper(env, pixels_only=True)
            logger.debug(f"spec: {env.spec}, Observation space: {env.observation_space}, action space: {env.action_space}")
        return env

    def setup(self, stage=None):
        # TODO: What should we be doing here for Gym environments?
        return super().setup(stage=stage)

    def prepare_data(self, *args, **kwargs):
        # TODO: What should we be doing here for Gym environments?
        super().prepare_data(*args, **kwargs)


    def train_dataloader(self, **kwargs) -> GymDataLoader:
        """Returns an ActiveDataLoader (GymDataLoader) for the training env.

        Returns:
            GymDataLoader: A DataLoader that also accepts actions.
        """
        dataset = self.gym_env_name
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        kwargs["num_workers"] = kwargs["batch_size"]
        self._train_loader = GymDataLoader(
            dataset,
            observe_pixels=not self.observe_state_directly,
            transforms=self.train_transforms,
            name="train",
            **kwargs
        )
        return self._train_loader

    def val_dataloader(self, **kwargs) -> GymDataLoader:
        """Returns an ActiveDataLoader (GymDataLoader) for the validation env.

        Returns:
            GymDataLoader: A DataLoader that also accepts actions.
        """
        dataset = self.gym_env_name
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        kwargs["num_workers"] = kwargs["batch_size"]
        self._val_loader = GymDataLoader(
            dataset,
            observe_pixels=not self.observe_state_directly,
            transforms=self.val_transforms,
            name="val",
            **kwargs
        )
        return self._val_loader

    def test_dataloader(self, **kwargs) -> GymDataLoader:
        """Returns an ActiveDataLoader (GymDataLoader) for the training env.

        Returns:
            GymDataLoader: A DataLoader that also accepts actions.
        """
        dataset = self.gym_env_name
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        kwargs["num_workers"] = kwargs["batch_size"]
        self._test_loader = GymDataLoader(
            dataset,
            observe_pixels=not self.observe_state_directly,
            transforms=self.test_transforms,
            name="test",
            **kwargs,
            
        )
        return self._test_loader

    @property
    def train_env(self) -> GymDataLoader:
        if self._train_loader is None:
            # TODO: This feels like it's got a code smell, but idk exactly how/why.
            raise RuntimeError("Can't call train_env before having called `train_dataloader`")
        return self._train_loader

    @property
    def val_env(self) -> GymDataLoader:
        return self._val_loader

    @property
    def test_env(self) -> GymDataLoader:
        return self._test_loader

    def send(self, actions: Tensor, environment: str) -> Tensor:
        if environment == "train":
            return self.train_env.send(actions)
        elif environment == "val":
            return self.val_env.send(actions)
        elif environment == "test":
            return self.test_env.send(actions)
        else:
            raise RuntimeError(
                f"Invalid environment '{environment}', (must be one of "
                f"'train', 'val' or 'test')"
            )

if __name__ == "__main__":
    RLSetting.main()
