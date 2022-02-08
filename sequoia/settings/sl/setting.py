from dataclasses import dataclass
from typing import ClassVar, Dict, List, Type, TypeVar

from pytorch_lightning import LightningDataModule
from simple_parsing import choice, list_field
from torch import Tensor

from sequoia.common.transforms import Transforms
from sequoia.settings import Setting
from sequoia.settings.base.environment import ActionType, ObservationType, RewardType

from .environment import PassiveEnvironment


@dataclass
class SLSetting(Setting[PassiveEnvironment[ObservationType, ActionType, RewardType]]):
    """Supervised Learning Setting.

    Core assuptions:
    - Current actions have no influence on future observations.
    - The environment gives back "dense feedback", (the 'reward' associated with all
      possible actions at each step, rather than a single action)

    For example, supervised learning is a Passive setting, since predicting a
    label has no effect on the reward you're given (the label) or on the next
    samples you observe.
    """

    @dataclass(frozen=True)
    class Observations(Setting.Observations):
        x: Tensor

    @dataclass(frozen=True)
    class Actions(Setting.Actions):
        pass

    @dataclass(frozen=True)
    class Rewards(Setting.Rewards):
        pass

    Environment: ClassVar[Type[PassiveEnvironment]] = PassiveEnvironment

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

    # Transforms to be applied to the observatons of the train/valid/test
    # environments.
    transforms: List[Transforms] = list_field()

    # Transforms to be applied to the training datasets.
    train_transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.three_channels)
    # Transforms to be applied to the validation datasets.
    val_transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.three_channels)
    # Transforms to be applied to the testing datasets.
    test_transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.three_channels)
    # Wether to drop the last batch (during training). Useful if you use batchnorm, to
    # avoid having an error when the batch_size is 1.
    drop_last: bool = False


SettingType = TypeVar("SettingType", bound=SLSetting)
