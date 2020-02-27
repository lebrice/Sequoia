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
from models.bases import BaseHParams


@dataclass  # type: ignore
class Experiment():
    """ Describes the parameters of an experimental setting. (ex: Mnist_iid, Mnist_continual, Cifar10, etc. etc.)
    
    To create a new experiment, subclass this class, and add/change what you
    need to customize.
    """
    name: ClassVar[str]
    dataset: Mnist = Mnist(iid=True)
    hparams: BaseHParams = BaseHParams()
    config: Config = Config()

    model: nn.Module = field(default=None, init=False)

    def __post_init__(self):
        """ Called after __init__, used to initialize all missing fields.
        
        You can use this method to initialize the fields that aren't parsed from
        the command-line, such as `model`, etc.        
        """ 
        pass

    @abstractmethod
    def run(self):
        raise NotImplementedError("Implement the 'run' method in a subclass.")
