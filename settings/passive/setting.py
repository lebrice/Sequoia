from dataclasses import dataclass

from settings.base.environment import (ActionType, ObservationType,
                                          RewardType)
from settings.base.setting import Setting
from settings.base.results import ResultsType
from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, MNISTDataModule)
from pytorch_lightning import LightningDataModule
from .environment import PassiveEnvironment
from typing import ClassVar, Dict, Type, TypeVar
from simple_parsing import choice

from torchvision.datasets import MNIST, FashionMNIST


@dataclass
class PassiveSetting(Setting[PassiveEnvironment[ObservationType, RewardType]]):
    """LightningDataModule for CL experiments.

    This greatly simplifies the whole data generation process.
    the train_dataloader, val_dataloader and test_dataloader methods are used
    to get the dataloaders of the current task.

    The current task can be set at the `current_task_id` attribute.

    TODO: Maybe add a way to 'wrap' another LightningDataModule?
    TODO: Change the base class from PassiveEnvironment to `ActiveEnvironment`
    and change the corresponding returned types. 
    """
    # TODO: rename/remove this, as it isn't used, and there could be some
    # confusion with the available_datasets in task-incremental and iid.
    # Also, since those are already LightningDataModules, what should we do?
    available_datasets: ClassVar[Dict[str, Type[LightningDataModule]]] = {
        "mnist": MNISTDataModule,
        "fashion_mnist": FashionMNISTDataModule,
        "cifar10": CIFAR10DataModule,
        "imagenet": ImagenetDataModule,
    }
    # Which setup / dataset to use.
    # The setups/dataset are implemented as `LightningDataModule`s. 
    dataset: str = choice(available_datasets.keys(), default="mnist")

SettingType = TypeVar("SettingType", bound=PassiveSetting)
