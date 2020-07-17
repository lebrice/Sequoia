
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import *
from typing import Callable

from torch.utils.data import DataLoader
from torchvision.transforms import Compose as ComposeBase
from torchvision.transforms import ToTensor

from datasets.data_utils import FixChannels
from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, LightningDataModule,
                                  MNISTDataModule)
from simple_parsing import choice, field, list_field, mutable_field
from utils.json_utils import Serializable

from .environment import (ActionType, ActiveEnvironment, EnvironmentBase,
                          ObservationType, PassiveEnvironment, RewardType)


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
        for e in cls:
            if type(e.value) == type(value):
                return e
        return super()._missing_(value)
        return cls[value]


class Compose(ComposeBase):
    def __init__(self, transforms: Sequence[Union[Transforms, Callable]]):
        self._transforms = transforms
        transforms = [
            t.value if isinstance(t, Transforms) else t
            for t in transforms
        ]
        super().__init__(transforms=transforms)
    
    def __contains__(self, other: Transforms) -> bool:
        return other in self._transforms
    
    def __iter__(self) -> Iterable[Callable]:
        yield from self.transforms


Loader = TypeVar("Loader", bound=DataLoader)

@dataclass
class ExperimentalSetting(LightningDataModule, Generic[Loader]):
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels)

    # TODO: Currently trying to find a way to specify the transforms from the command-line.
    # As a consequence, don't try to set these from the command-line for now.
    train_transforms: List[Transforms] = list_field()
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
