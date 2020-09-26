""" IDEA: Not sure how to best do this really, but we want to make the
pl_bolts datamodules into IID settings directly. """

from pl_bolts.datamodules import (MNISTDataModule, CIFAR10DataModule,
                                  FashionMNISTDataModule, ImagenetDataModule,
                                  STL10DataModule)
from settings import Setting, IIDSetting
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MnistSetting(MNISTDataModule, IIDSetting):
    # TODO: Fix this mess. 
    data_dir: Path = Path("data")
    val_split: int = 5000
    num_workers: int = 16
    def __post_init__(self, *args, **kwargs):
        MNISTDataModule.__init__(self,
            data_dir=self.data_dir,
            val_split=self.val_split,
            num_workers=self.num_workers,
        )
        super().__post_init__(*args, **kwargs)


@dataclass
class FashionMNISTSetting(FashionMNISTDataModule, IIDSetting):
    pass


@dataclass
class STL10Setting(STL10DataModule, IIDSetting):
    data_dir: str = None
    unlabeled_val_split: int = 5000
    train_val_split: int = 500
    num_workers: int = 16
    batch_size: int = 32