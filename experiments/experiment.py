from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import *
from typing import ClassVar

import torch
from simple_parsing import field
from torch import nn

from config import Config
from datasets import Dataset
from datasets.mnist import Mnist

from models.classifier import Classifier

@dataclass
class Experiment:
    """ Describes the parameters of an experimental setting.
    
    (ex: Mnist_iid, Mnist_continual, Cifar10, etc. etc.)
    
    To create a new experiment, subclass this class, and add/change what you
    need to customize.

    TODO: Maybe add some code for saving/restoring experiments here?
    """
    # Dataset and preprocessing settings.
    dataset: Dataset = Mnist()
    # Model Hyperparameters 
    hparams: Classifier.HParams = Classifier.HParams()
    # Settings related to the experimental setup (cuda, log_dir, etc.).
    config: Config = Config()
    
    model: nn.Module = field(default=None, init=False)

    class_incremental: bool = False

    def __post_init__(self):
        """ Called after __init__, used to initialize all missing fields.
        
        You can use this method to initialize the fields that aren't parsed from
        the command-line, such as `model`, etc.        
        """ 
        pass

    def run(self):
        raise NotImplementedError("Implement the 'run' method in a subclass.")
