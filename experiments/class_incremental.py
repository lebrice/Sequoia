from dataclasses import dataclass
from typing import Iterable, List, Union

from simple_parsing import choice, field, mutable_field, subparsers
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.datasets as v_datasets
from common.losses import LossInfo
from config import Config, ClassIncrementalConfig
from datasets.dataset import make_class_incremental
from experiments.experiment import Experiment
from experiments.iid import IID
from models.classifier import Classifier
from models.ss_classifier import SelfSupervisedClassifier

from datasets.dataset import TaskConfig

@dataclass
class ClassIncremental(IID):
    class_incremental: ClassIncrementalConfig = ClassIncrementalConfig(True)
    online: bool = False  # wether or not to perform a single epoch of training.

    def __post_init__(self):
        super().__post_init__()
    
    def load(self):
        if self.online:
            self.hparams.epochs = 1

        self.dataset.load(self.config)
        assert self.dataset.train
        assert self.dataset.valid
        self.dataset.train_tasks = make_class_incremental(self.dataset.train, self.class_incremental)
        self.dataset.valid_tasks = make_class_incremental(self.dataset.valid, self.class_incremental)

        if self.config.debug:
            print("Class Incremental Setup:")
            print("Training tasks:")
            print(*self.dataset.train_tasks, sep="\n")
            print("Validation tasks:")
            print(*self.dataset.valid_tasks, sep="\n")
        
        self.save_images_for_each_task(
            self.dataset.train,
            self.dataset.train_tasks,
            prefix="train_",
        )
        self.save_images_for_each_task(
            self.dataset.valid,
            self.dataset.valid_tasks,
            prefix="valid_",
        )
        super().load()

    
    def save_images_for_each_task(self,
                                  dataset: v_datasets.MNIST,
                                  tasks: List[TaskConfig],
                                  prefix: str="",
                                  n: int=64):
        n = 64
        for task in tasks:
            start = task.start_index
            stop = start + n
            sample = dataset.data[start: start+n].view(n, *self.dataset.x_shape).float()
            save_image(sample, self.samples_dir / f"{prefix}task_{task.id}.png")
