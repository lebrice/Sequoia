""" Defines the TraditionalSLSetting, as a variant of the TaskIncremental setting with
only one task.
"""
from dataclasses import dataclass
from typing import (Callable, ClassVar, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union)
import itertools
import tqdm
from torch import Tensor

from sequoia.common.loss import Loss
from sequoia.common.metrics import Metrics
from sequoia.common.config import Config
from sequoia.settings.base import Results
from sequoia.utils.utils import constant, dict_union

# TODO: Re-arrange the 'multiple-inheritance' with domain-incremental and
# task-incremental, this might not be 100% accurate, as the "IID" you get from
# moving down from domain-incremental (+ only one task) might not be exactly the same as
# the one you get form TaskIncremental (+ only one task)
from sequoia.settings.sl.multi_task import MultiTaskSLSetting
# from sequoia.settings.sl.domain_incremental import DomainIncrementalSLSetting
from .results import IIDResults


# TODO: IDEA: Add the pytorch lightning datamodules in the list of
# 'available datasets' for the IID setting, and make sure that it doesn't mess
# up the methods in the parents (train/val loop, dataloader construction, etc.)
# IDEA: Maybe overwrite the 'train/val/test_dataloader' methods on the setting
# and when the chosen dataset is a LightnignDataModule, then just return the
# result from the corresponding method on the LightningDataModule, rather than
# from super().
# from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
#                                   ImagenetDataModule, MNISTDataModule)


@dataclass
class TraditionalSLSetting(MultiTaskSLSetting):
    """Your 'usual' supervised learning Setting, where the samples are i.i.d.
    
    This Setting is slightly different than the others, in that it can be recovered in
    *two* different ways:
    - As a variant of Task-Incremental learning, but where there is only one task;
    - As a variant of Domain-Incremental learning, but where there is only one task.
    """
    Results: ClassVar[Type[Results]] = IIDResults

    # Number of tasks, held constant, since this is an IID setting.
    # TODO: This setting could also be a 'multi-task' Supervised learning setting, if
    # this were to be left 'floating'.
    nb_tasks: int = constant(1)

    # increment: Union[int, List[int]] = constant(None)
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


SettingType = TypeVar("SettingType", bound=TraditionalSLSetting)


if __name__ == "__main__":
    TraditionalSLSetting.main()
