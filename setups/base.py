
from dataclasses import dataclass
from enum import Enum
from typing import *

from torchvision.transforms import Compose, ToTensor

from datasets.data_utils import FixChannels
from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, LightningDataModule,
                                  MNISTDataModule)
from simple_parsing import choice, field, list_field, mutable_field
from utils.json_utils import Serializable

from .environment import (ActionType, ActiveEnvironment, ObservationType,
                          PassiveEnvironment, RewardType, EnvironmentBase)
from abc import abstractmethod

class Transforms(Enum):
    """ Enum of possible transforms. 
    TODO: Maybe use this to create a customizable input pipeline (with the Simclr MoCo/etc augments?)
    """
    fix_channels = FixChannels()
    to_tensor = ToTensor()

    def __mult__(self, other: "Transforms"):
        # TODO: maybe use multiplication as composition?
        return NotImplemented
    @classmethod
    def _missing_(cls, value: Any):
        return cls[value]
        

def compose(transforms) -> Compose:
    if isinstance(transforms, (list, tuple)):
        if len(transforms) == 1:
            return transforms[0]
        elif len(transforms) > 1:
            return Compose(transforms)
    return transforms
from torch.utils.data import DataLoader
Loader = TypeVar("Loader", bound=DataLoader)

@dataclass
class ExperimentalSetting(LightningDataModule, Generic[Loader]):
        
    @dataclass
    class Config(Serializable):
        """
        Represents all the configuration options related to a Setup. (LightningModule)
        """
        # Default transform to use. (not settable by the command-line.)
        _default_transform: ClassVar[Callable] = Compose([
            ToTensor(),
            FixChannels(),
        ])

        # TODO: Currently trying to find a way to specify the transforms from the command-line.
        # As a consequence, don't try to set these from the command-line for now.
        transforms: List[Transforms] = field(default=_default_transform, to_dict=False)
        train_transforms: List[Transforms] = list_field(to_dict=False)
        valid_transforms: List[Transforms] = list_field(to_dict=False)
        test_transforms: List[Transforms] = list_field(to_dict=False)

        def __post_init__(self):
            self.transforms = compose(self.transforms)
            self.train_transforms = compose(self.train_transforms) or self.transforms
            self.valid_transforms = compose(self.valid_transforms) or self.transforms
            self.test_transforms = compose(self.test_transforms) or self.transforms

    # Configuration options for the environment / setup / datasets.
    config: Config = mutable_field(Config)

    def __init__(self, config: Config):
        """Creates a new Environment / setup.

        Args:
            config (Config): Dataclass used for configuration.
        """
        super().__init__(
            train_transforms=config.train_transforms,
            val_transforms=config.val_transforms,
            test_transforms=config.test_transforms,
        )
        self.config: "ExperimentalSetup.Config" = config
    
    @abstractmethod
    def train_dataloader(self, *args, **kwargs) -> Loader:
        return super().train_dataloader(*args, **kwargs)
    
    @abstractmethod
    def val_dataloader(self, *args, **kwargs) -> Loader:
        return super().val_dataloader(*args, **kwargs)
    
    @abstractmethod
    def test_dataloader(self, *args, **kwargs) -> Loader:
        return super().test_dataloader(*args, **kwargs)


@dataclass
class ActiveSetup(ExperimentalSetting[ActiveEnvironment[ObservationType, ActionType, RewardType]]):
    """LightningDataModule for CL experiments.

    This greatly simplifies the whole data generation process.
    the train_dataloader, val_dataloader and test_dataloader methods are used
    to get the dataloaders of the current task.

    The current task can be set at the `current_task_id` attribute.

    TODO: Maybe add a way to 'wrap' another LightningDataModule?
    TODO: Change the base class from PassiveEnvironment to `ActiveEnvironment`
    and change the corresponding returned types. 
    """
    @dataclass
    class Config(ExperimentalSetting.Config):
        """ Configuration options for an "active" experimental setup.
        TODO: Add RL environments here? or just some active learning stuff?
        """
    
    config: Config = mutable_field(Config)

@dataclass
class PassiveSetting(ExperimentalSetting[PassiveEnvironment[ObservationType, RewardType]]):
    """LightningDataModule for CL experiments.

    This greatly simplifies the whole data generation process.
    the train_dataloader, val_dataloader and test_dataloader methods are used
    to get the dataloaders of the current task.

    The current task can be set at the `current_task_id` attribute.

    TODO: Maybe add a way to 'wrap' another LightningDataModule?
    TODO: Change the base class from PassiveEnvironment to `ActiveEnvironment`
    and change the corresponding returned types. 
    """
    @dataclass
    class Config(ExperimentalSetting.Config):
        """ Configuration options for a passive experimental setup. """
        available_environments: ClassVar[Dict[str, Type[LightningDataModule]]] = {
            "mnist": MNISTDataModule,
            "fashion_mnist": FashionMNISTDataModule,
            "cifar10": CIFAR10DataModule,
            "imagenet": ImagenetDataModule,
        }
        # Which setup / dataset to use.
        # The setups/dataset are implemented as `LightningDataModule`s. 
        dataset: str = choice(available_environments.keys(), default="mnist")

    config: Config = mutable_field(Config)