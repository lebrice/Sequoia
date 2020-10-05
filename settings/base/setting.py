""" This module defines the `Setting` class, an ML "problem" to solve. 

The `Setting` class is an abstract base class which should represent the most
general learning setting imaginable, i.e. with the fewest assumptions about the
data, the environment, the agent, etc.


The Setting class is currently loosely based on the `LightningDataModule` class
from pytorch-lightning, with the goal of having an `IIDSetting` node somewhere
in the tree, which would be totally interchangeable with existing datamodules
from pytorch-lightning.

The hope is that by staying close to that API, we can make it easier for people
to adopt the repo, and also, if possible, directly reuse existing models from
pytorch-lightning.

See: [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)  
See: [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html)

"""
import inspect
import os
import shlex
from abc import abstractmethod
from argparse import Namespace
from collections import OrderedDict
from dataclasses import InitVar, dataclass, fields, is_dataclass
from inspect import getsourcefile, isclass
from functools import partial
from pathlib import Path
from typing import *

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.datamodule import _DataModuleWrapper
from simple_parsing import (ArgumentParser, Serializable, list_field,
                            mutable_field, subparsers)
from torch.utils.data import DataLoader

from common.config import Config
from common.loss import Loss
from common.transforms import Compose, Transforms, Transform, SplitBatch
from utils import Parseable, camel_case, dict_union, get_logger, remove_suffix

from .results import Results
from .setting_meta import SettingMeta
from .environment import Environment, Observations, Actions, Rewards

logger = get_logger(__file__)

EnvironmentType = TypeVar("EnvironmentType", bound=Environment)
SettingType = TypeVar("SettingType", bound="Setting")

from ..setting_abc import SettingABC
from ..method_abc import MethodABC

@dataclass
class Setting(SettingABC,
              Generic[EnvironmentType],
              Serializable,
              Parseable,
              metaclass=SettingMeta):
    """ Base class for all research settings in ML: Root node of the tree. 

    A 'setting' is loosely defined here as a learning problem with a specific
    set of assumptions, restrictions, and an evaluation procedure.
    
    For example, Reinforcement Learning is a type of Setting in which we assume
    that an Agent is able to observe an environment, take actions upon it, and 
    receive rewards back from the environment. Some of the assumptions include
    that the reward is dependant on the action taken, and that the actions have
    an impact on the environment's state (and on the next observations the agent
    will receive). The evaluation procedure consists in trying to maximize the
    reward obtained from an environment over a given number of steps.
        
    This 'Setting' class should ideally represent the most general learning
    problem imaginable, with almost no assumptions about the data or evaluation
    procedure.

    This is a dataclass. Its attributes are can also be used as command-line
    arguments using `simple_parsing`.
    """
    ## ---------- Class Variables ------------- 
    ## Fields in this block are class attributes. They don't create command-line
    ## arguments.
    
    # Type of Observations that the dataloaders (a.k.a. "environments") will
    # produce for this type of Setting.
    Observations: ClassVar[Type[Observations]] = Observations
    # Type of Actions that the dataloaders (a.k.a. "environments") will receive
    # through their `send` method, for this type of Setting.
    Actions: ClassVar[Type[Actions]] = Actions
    # Type of Rewards that the dataloaders (a.k.a. "environments") will return
    # after receiving an action, for this type of Setting.
    Rewards: ClassVar[Type[Rewards]] = Rewards
    
    # The type of Results that are given back when a method is applied on this
    # Setting. The `Results` class basically defines the 'evaluation metric' for
    # a given type of setting. See the `Results` class for more info.
    Results: ClassVar[Type[Results]] = Results
    
    
    ##
    ##   -------------
    
    # Transforms to be used. When no value is given for 
    # `[train/val/test]_transforms`, this value is used as a default.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.three_channels)
    # Transforms to be applied to the training datasets. When unset, the value
    # of `transforms` is used.
    train_transforms: List[Transforms] = list_field()
    # Transforms to be applied to the validation datasets. When unset, the value
    # of `transforms` is used.
    val_transforms: List[Transforms] = list_field()
    # Transforms to be applied to the testing datasets. When unset, the value
    # of `transforms` is used.
    test_transforms: List[Transforms] = list_field()
   

    # Fraction of training data to use to create the validation set.
    val_fraction: float = 0.2

    # TODO: Add support for semi-supervised training.
    # Fraction of the dataset that is labeled.
    labeled_data_fraction: int = 1.0
    # Number of labeled examples.
    n_labeled_examples: Optional[int] = None

    # These should be set by all settings, not from the command-line. Hence we
    # mark them as InitVars, which means they should be passed to __post_init__.
    obs_shape: Tuple[int, ...] = ()
    action_shape: Tuple[int, ...] = ()
    reward_shape: Tuple[int, ...] = ()

    def __post_init__(self,
                      obs_shape: Tuple[int, ...] = (),
                      action_shape: Tuple[int, ...] = (),
                      reward_shape: Tuple[int, ...] = ()):
        """ Initializes the fields of the setting that weren't set from the
        command-line.
        """
        logger.debug(f"__post_init__ of Setting")
        # Actually compose the list of Transforms or callables into a single transform.
        # self.transforms: Compose = Compose(self.transforms)
        # self.train_transforms: Compose = Compose(self.train_transforms())
        # self.val_transforms: Compose = Compose(self.val_transforms())
        # self.test_transforms: Compose = Compose(self.test_transforms())
        # TODO: Adds an additional transform that splits stuff into Observations
        # and Rewards.
        # IDEA: Add a `BatchTransform` class to mark the transforms which can
        # operate on batch objects rather than just on tensors. Use
        # __init_subclass__ in that class to wrap the __call__ method
        # of the transforms and get the tensor from the observation
        # automatically.
        self.transforms: Compose = Compose(self.transforms)
        self.train_transforms: Compose = Compose(self.train_transforms or self.transforms)
        self.val_transforms: Compose = Compose(self.val_transforms or self.transforms)
        self.test_transforms: Compose = Compose(self.test_transforms or self.transforms)
        print(type(self).__bases__)
        
        LightningDataModule.__init__(self,
            train_transforms=self.train_transforms,
            val_transforms=self.val_transforms,
            test_transforms=self.test_transforms,
        )
        # Transform that will split batches into Observations and Rewards.
        self.split_batch_transform = SplitBatch(
            observation_type=self.Observations,
            reward_type=self.Rewards
        )

        self.obs_shape = self.obs_shape or obs_shape
        self.action_shape = self.action_shape or action_shape
        self.reward_shape = self.reward_shape or reward_shape

        logger.debug(f"Transforms: {self.transforms}")
        if obs_shape and self.transforms:
            # TODO: Testing out an idea: letting the transforms tell us how
            # they change the shape of the image.
            logger.debug(f"Obs shape before transforms: {obs_shape}")
            self.obs_shape: Tuple[int, ...] = self.transforms.shape_change(obs_shape)
            logger.debug(f"Obs shape after transforms: {self.obs_shape}")

        self.dataloader_kwargs: Dict[str, Any] = {}

        if self.obs_shape and not self.dims:
            self.dims = self.obs_shape

        # This should probably be set on `self` inside of `apply` call.
        # TODO: It's a bit confusing to also have a `config` attribute on the
        # Setting. Might want to change this a bit.
        self.config: Config = None

    @abstractmethod
    def apply(self, method: MethodABC, config: Config) -> Results:
        pass

    # Transforms that apply on Observation objects (the whole batch).
    # TODO: (@lebrice): This is still a bit wonky, need to design this better.
    def train_batch_transforms(self) -> List[Callable]:
        return [self.split_batch_transform]
    def valid_batch_transforms(self) -> List[Callable]:
        return self.train_batch_transforms()
    def test_batch_transforms(self) -> List[Callable]:
        return self.valid_batch_transforms()
    
    @classmethod
    def main(cls, argv: Optional[Union[str, List[str]]]=None) -> Results:
        from main import Experiment
        experiment: Experiment
        # Create the Setting object from the command-line:
        setting = cls.from_args(argv)
        # Then create the 'Experiment' from the command-line, which makes it
        # possible to choose between all the methods.
        experiment = Experiment.from_args(argv)
        # fix the setting attribute to be the one parsed above.
        experiment.setting = setting
        results: ResultsType = experiment.launch(argv)
        return results

    def apply_all(self, argv: Union[str, List[str]] = None) -> Dict[Type["Method"], Results]:
        applicable_methods = self.get_applicable_methods()
        from methods import Method
        all_results: Dict[Type[Method], Results] = OrderedDict()
        config = Config.from_args(argv)
        for method_type in applicable_methods:
            method = method_type.from_args(argv)
            results = self.apply(method, config)
            all_results[method_type] = results
        logger.info(f"All results for setting of type {type(self)}:")
        logger.info({
            method.get_name(): (results.get_metric() if results else "crashed")
            for method, results in all_results.items()
        })
        return all_results

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        if not is_dataclass(cls):
            return super().add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False,)
        dest: str = cls.__qualname__
        parser.add_arguments(cls, dest=dest)
        return parser

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
        dest: str = cls.__qualname__
        if hasattr(args, dest):
            instance = args.dest
            assert not kwargs, f"kwargs: {kwargs}"
            return instance
        return super().from_argparse_args(args=args, **kwargs)

    @classmethod
    def get_path_to_source_file(cls: Type) -> Path:
        from utils.utils import get_path_to_source_file
        return get_path_to_source_file(cls)
