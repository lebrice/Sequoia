from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple, Type

import gym
from pytorch_lightning import Trainer
from torch import Tensor
from torch.utils.data import DataLoader

from common.config import Config
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
    pass


@dataclass
class RLSetting(ActiveSetting[Tensor, Tensor, Tensor], Pickleable):
    """
    """
    results_class: ClassVar[Type[Results]] = RLResults
    # Class variable holding all the available datasets.
    available_datasets: ClassVar[Dict[str, str]] = {
        "cartpole": "CartPole-v0",
    }
    observe_state_directly: bool = False
    dataset: str = choice(available_datasets, default="CartPole-v0")

    # Transformations to use. See the Transforms enum for the available values.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels)
    
    # Starting with batch size fixed to 2 for now.
    # batch_size: int = 2

    def __post_init__(self):
        """Creates a new RL environment / setup. """
        # FIXME: (@lebrice) This is a bit annoying to have to do, but whatever.
        # The idea is that we want to set some 'dims' attributes so that methods
        # can know what the observations / actions / rewards will look like.
        # Btw this also assumes that the shapes don't change between train, val
        # and test (which seems very reasonable for now).
        train_env = GymDataset(
            env=self.gym_env_name,
            observe_pixels=not self.observe_state_directly,
        )
        obs_shape: Tuple[int, ...] = train_env.observation_space.shape
        action_shape: Tuple[int, ...] = train_env.action_space.shape or (1,)
        # NOTE: We assume scalar rewards for now.
        reward_shape: Tuple[int, ...] = (1,)

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

        self.action_space = train_env.action_space
        self.observation_space = train_env.observation_space
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
            # TODO: Choose either.
            datamodule=self,
            # test_dataloaders=self.test_dataloader(),
            verbose=True,
        )
        assert test_outputs, f"Test outputs should be produced: {test_outputs}"
        
        model = method.model
        from methods.models import Model
        if isinstance(model, Model):
            hparams = model.hp
        else:
            hparams = model.hparams
        return self.results_class(
            hparams=hparams,
            # test_loss=sum(task_losses),
            # task_losses=task_losses,
        )

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
            **kwargs
        )
        return self._test_loader

    @property
    def train_env(self) -> GymDataLoader:
        if self._train_loader is None:
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
