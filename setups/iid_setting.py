from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type,
                    Union)

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transform_lib
from torchvision.datasets import (CIFAR10, CIFAR100, MNIST, FashionMNIST,
                                  ImageNet, VisionDataset)
from torchvision.transforms import Compose, Resize, ToTensor

import pl_bolts
from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, LightningDataModule,
                                  MNISTDataModule, SSLImagenetDataModule)
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from simple_parsing import choice, field, list_field
from utils.json_utils import Serializable, encode, register_decoding_fn

from datasets.data_utils import FixChannels, keep_in_memory, train_valid_split

# data_dir = Path("data")
# data_module = MNISTDataModule(data_dir, val_split=5000, num_workers=16, normalize=False)
# raise NotImplementedError("TODO")
from .base import PassiveSetting, ObservationType, RewardType
from .transforms import Transforms, Compose
from common.dims import Dims
from .cl import ClassIncrementalSetting
from typing import TypeVar
T = TypeVar("T")

from simple_parsing import field

def constant(v: T, **kwargs) -> T:
    return field(default=v, init=False, **kwargs)



@dataclass
class IIDSetting(ClassIncrementalSetting):
    """ Normal IID Setting.
    
    This is implemented quite simply as taking all the datasets from Continuum
    and setting the number of tasks to 1 (which makes sense!).

    TODO: Mark the relevant options as constant somehow, so they can't get set
    from the command-line.
    """
    # Held constant, since this is an IID setting.
    nb_tasks: int = constant(1)
    increment: Union[int, List[int]] = constant(None)
    # A different task size applied only for the first task.
    # Desactivated if `increment` is a list.
    initial_increment: int = constant(None)
    # An optional custom class order, used for NC.
    class_order: Optional[List[int]] = constant(None)
    # Either number of classes per task, or a list specifying for
    # every task the amount of new classes (defaults to the value of
    # `increment`).
    test_increment: Optional[Union[List[int], int]] = constant(None)
    # A different task size applied only for the first test task.
    # Desactivated if `test_increment` is a list. Defaults to the
    # value of `initial_increment`.
    test_initial_increment: Optional[int] = constant(None)
    # An optional custom class order for testing, used for NC.
    # Defaults to the value of `class_order`.
    test_class_order: Optional[List[int]] = constant(None)
