from pathlib import Path

from continuum import ClassIncremental, split_train_val
from continuum.datasets import MNIST
from torch.utils.data import DataLoader

import pl_bolts
from pl_bolts.datamodules import MNISTDataModule, LightningDataModule

clloader = ClassIncremental(
    MNIST("my/data/path", download=True),
    increment=1,
    initial_increment=5,
    train=True  # a different loader for test
)

print(f"Number of classes: {clloader.nb_classes}.")
print(f"Number of tasks: {clloader.nb_tasks}.")

for task_id, train_dataset in enumerate(clloader):
    train_dataset, val_dataset = split_train_val(train_dataset, val_split=0.1)
    train_loader = DataLoader(train_dataset)
    val_loader = DataLoader(val_dataset)

from .environment_base import PassiveEnvironment

class CLEnvironment(LightningDataModule, PassiveEnvironment):
    pass

