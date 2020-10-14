from dataclasses import dataclass
from typing import ClassVar, Dict, Type, TypeVar

from pytorch_lightning import LightningDataModule
from settings.base.environment import ActionType, ObservationType, RewardType
from settings.base.results import ResultsType
from settings import Setting
from simple_parsing import choice
from torchvision.datasets import MNIST, FashionMNIST

from .passive_environment import PassiveEnvironment

# from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
#                                   ImagenetDataModule, MNISTDataModule)

@dataclass
class PassiveSetting(Setting[PassiveEnvironment[ObservationType, ActionType, RewardType]]):
    """Setting where actions have no influence on future observations. 

    For example, supervised learning is a Passive setting, since predicting a
    label has no effect on the reward you're given (the label) or on the next
    samples you observe.
    """
    # TODO: rename/remove this, as it isn't used, and there could be some
    # confusion with the available_datasets in task-incremental and iid.
    # Also, since those are already LightningDataModules, what should we do?
    available_datasets: ClassVar[Dict[str, Type[LightningDataModule]]] = {
        # "mnist": MNISTDataModule,
        # "fashion_mnist": FashionMNISTDataModule,
        # "cifar10": CIFAR10DataModule,
        # "imagenet": ImagenetDataModule,
    }
    # Which setup / dataset to use.
    # The setups/dataset are implemented as `LightningDataModule`s. 
    dataset: str = choice(available_datasets.keys(), default="mnist")

SettingType = TypeVar("SettingType", bound=PassiveSetting)
