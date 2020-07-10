import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, LightningDataModule,
                                  MNISTDataModule, SSLImagenetDataModule)
from pl_bolts.models import LogisticRegression
from pl_bolts.models.self_supervised import CPCV2, SimCLR
from pl_bolts.models.self_supervised.simclr import (SimCLREvalDataTransform,
                                                    SimCLRTrainDataTransform)
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import metrics, seed_everything
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms
from torchvision.datasets import MNIST

from config.config import Config as ConfigBase
from datasets import DatasetConfig
from models.pretrained_model import get_pretrained_encoder
from simple_parsing import (ArgumentParser, Serializable, choice, field,
                            mutable_field)
from utils.logging_utils import get_logger

logger = get_logger(__file__)
from models.classifier import Classifier


@dataclass
class Experiment(Serializable):
    
    @dataclass
    class Config(ConfigBase):
        pass
    
    @dataclass
    class State(Serializable):
        """ object which tracks the state of an experiment. 
        
        TODO: Might not be useful anymore.
        TODO: (somewhat unrelated) What does the `experiment` kwarg to WandbLogger do?
        """
        pass


    # HyperParameters of the model/experiment.
    hparams: Classifier.HParams = mutable_field(Classifier.HParams)
    # Configuration options for an experiment (log_dir, etc.)
    config: Config = mutable_field(Config)

    state: State = mutable_field(State, repr=False)

    def run(self):
        raise RuntimeError("Implement your own run method in a subclass!")
        model = Classifier(hparams=self.hparams, config=self.config)
        trainer = self.config.make_trainer()
        trainer.fit(model)
        

    def launch(self):
        print("Launching experiment.")
        if self.config.debug:
            print(self.dumps_yaml(indent=1))
        self.run()
        # MnistModel.load_from_checkpoint("test")
        # trainer.test(model)
        # trainer.save_checkpoint("test")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Experiment, "experiment")
    args = parser.parse_args()
    experiment: Experiment = args.experiment
    experiment.launch()
