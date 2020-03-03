from dataclasses import dataclass
from typing import Iterable, Union, List

from simple_parsing import choice, field, subparsers
from torch.utils.data import DataLoader

from common.losses import LossInfo
from config import Config

from models.classifier import Classifier
from models.ss_classifier import SelfSupervisedClassifier
from experiments.experiment import Experiment

@dataclass 
class ClassIncrementalConfig:
    class_incremental: bool = False  # train in a class-incremental fashion.
    n_classes_per_task: int = 2      # Number of classes per task.
    # Wether to sort out the classes in the class_incremental setting.
    random_class_ordering: bool = False


@dataclass
class ClassIncremental(Experiment):
    name: str = "class_incremental"
    config: Config = Config(class_incremental=True)
    online: bool = False  # wether or not to perform a single epoch of training.

    def __post_init__(self):
        if self.online:
            self.hparams.epochs = 1
            self.name += "_online"
        super().__post_init__()

    def train_iter(self, epoch: int, dataloader: DataLoader) -> Iterable[LossInfo]:
        for train_loss in super().train_iter(epoch, dataloader):
            yield train_loss

    def make_plots_for_epoch(self, epoch: int, train_batch_losses: List[LossInfo], valid_batch_losses: List[LossInfo]):
        print(epoch)
        # print(train_batch_losses)
        exit()