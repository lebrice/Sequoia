
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import *
from typing import Callable

from torch.utils.data import DataLoader
from torchvision.transforms import Compose as ComposeBase
from torchvision.transforms import ToTensor

from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, LightningDataModule,
                                  MNISTDataModule)
from simple_parsing import choice, field, list_field, mutable_field
from utils.json_utils import Serializable

from .environment import (ActionType, ActiveEnvironment, EnvironmentBase,
                          ObservationType, PassiveEnvironment, RewardType)
from torch import Tensor


Loader = TypeVar("Loader", bound=DataLoader)
from .transforms import Transforms, Compose

@dataclass
class ExperimentalSetting(LightningDataModule, Generic[Loader]):
    """Extends LightningDataModule to allow setting the transforms and options
    from the command-line.

    This class is Generic, which allows us to pass a different `Loader` type, 
    which should be the type of dataloader returned by the `train_dataloader`,
    `val_dataloader` and `test_dataloader` methods.

    Args:
        LightningDataModule ([type]): [description]
        Generic ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Transforms to be used.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels)

    # TODO: Currently trying to find a way to specify the transforms from the command-line.
    # As a consequence, don't try to set these from the command-line for now.
    train_transforms: List[Transforms] = list_field()

    # TODO: These two aren't being used atm (at least not in ClassIncremental), 
    # since the CL Loader from Continuum only takes in common_transforms and
    # train_transforms.
    val_transforms: List[Transforms] = list_field()
    test_transforms: List[Transforms] = list_field()

    def __post_init__(self):
        """Creates a new Environment / setup.

        Args:
            options (Options): Dataclass used for configuration.
        """
        train_transforms: Callable = Compose(self.train_transforms or self.transforms)
        val_transforms: Callable = Compose(self.val_transforms or self.transforms)
        test_transforms: Callable = Compose(self.test_transforms or self.transforms)
        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
        )

    @abstractmethod
    def train_dataloader(self, **kwargs) -> Loader:
        return super().train_dataloader(**kwargs)
    
    @abstractmethod
    def val_dataloader(self, **kwargs) -> Loader:
        return super().val_dataloader(**kwargs)
    
    @abstractmethod
    def test_dataloader(self, **kwargs) -> Loader:
        return super().test_dataloader(**kwargs)


@dataclass
class ActiveSetup(ExperimentalSetting[ActiveEnvironment[ObservationType, ActionType, RewardType]]):
    """LightningDataModule for an 'active' setting.
    
    TODO: Use this for something like RL or Active Learning.
    """

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
    available_environments: ClassVar[Dict[str, Type[LightningDataModule]]] = {
        "mnist": MNISTDataModule,
        "fashion_mnist": FashionMNISTDataModule,
        "cifar10": CIFAR10DataModule,
        "imagenet": ImagenetDataModule,
    }
    # Which setup / dataset to use.
    # The setups/dataset are implemented as `LightningDataModule`s. 
    dataset: str = choice(available_environments.keys(), default="mnist")
