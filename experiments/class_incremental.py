from dataclasses import dataclass
from typing import Iterable, List, Union

from simple_parsing import choice, field, mutable_field, subparsers
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.datasets as v_datasets
from common.losses import LossInfo
from config import Config
from datasets.dataset import make_class_incremental
from experiments.experiment import Experiment
from experiments.iid import IID
from models.classifier import Classifier
from models.ss_classifier import SelfSupervisedClassifier

from datasets.dataset import TaskConfig


@dataclass
class ClassIncremental(IID):
    n_classes_per_task: int = 2      # Number of classes per task.
    # Wether to sort out the classes in the class_incremental setting.
    random_class_ordering: bool = False
    online: bool = False  # wether or not to perform a single epoch of training.
    
    def __post_init__(self):
        print("called postinit of ClassIncremental")
        super().__post_init__()
        if self.online:
            self.hparams.epochs = 1
        if self.config.debug:
            print("Class Incremental Setup:")
            print("Training tasks:")
            print("\t", *self.dataset.train_tasks, sep="\n\t")
            print("Validation tasks:")
            print("\t", *self.dataset.valid_tasks, sep="\n\t")
    
    def load(self):
        # TODO: Clean-up this mechanism by having the dataset.load() method take in classes to use and return the dataloaders.
        self.dataset.load(self.config)

        assert self.dataset.train
        assert self.dataset.valid
        
        self.dataset.train_tasks = make_class_incremental(
            self.dataset.train,
            n_classes_per_task=self.n_classes_per_task,
            random_ordering=self.random_class_ordering,
        )
        self.dataset.valid_tasks = make_class_incremental(
            self.dataset.valid,
            n_classes_per_task=self.n_classes_per_task,
            random_ordering=self.random_class_ordering,
        )        
        super().load()

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

        if isinstance(self.model, SelfSupervisedClassifier):
            print("Auxiliary tasks:")
            for task in self.model.tasks:
                self.log(f"{task.name} coefficient: ", task.coefficient, once=True, always_print=True)


    def save_images_for_each_task(self,
                                  dataset: v_datasets.MNIST,
                                  tasks: List[TaskConfig],
                                  prefix: str="",
                                  n: int=64):
        n = 64
        for i, task in enumerate(tasks):
            start = task.start_index
            stop = start + n
            sample = dataset.data[start: start+n].view(n, *self.dataset.x_shape).float()
            save_image(sample, self.samples_dir / f"{prefix}task_{i}.png")
