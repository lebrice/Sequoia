from dataclasses import dataclass
from typing import Iterable

from simple_parsing import choice, field
from torch.utils.data import DataLoader

from common.losses import LossInfo
from config import Config
from experiments.self_supervised import SelfSupervised


@dataclass
class ClassIncremental(SelfSupervised):
    config: Config = Config(class_incremental=True)
    
    def train_iter(self, epoch: int, dataloader: DataLoader) -> Iterable[LossInfo]:
        for train_loss in super().train_iter(epoch, dataloader):
            print(train_loss.tensors.keys())
            exit()
            yield train_loss
