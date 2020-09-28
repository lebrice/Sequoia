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
from torch.utils.data import DataLoader

from common.config import Config
from common.loss import Loss
from common.transforms import Compose, Transforms
from simple_parsing import (ArgumentParser, Serializable, list_field,
                            mutable_field, subparsers)
from utils import Parseable, camel_case, dict_union, get_logger, remove_suffix

from .results import Results

logger = get_logger(__file__)

Loader = TypeVar("Loader", bound=DataLoader)
ResultsType = TypeVar("ResultsType", bound=Results)
SettingType = TypeVar("SettingType", bound="Setting")


class SettingMeta(_DataModuleWrapper, Type["Setting"]):
    """ Metaclass for the nodes in the Setting inheritance tree.
    
    Might remove this. Was experimenting with using this to create class
    properties for each Setting.

    TODO: A little while back I noticed some strange behaviour when trying
    to create a Setting class (either manually or through the command-line), and
    I attributed it to PL adding a `_DataModuleWrapper` metaclass to
    `LightningDataModule`, which seemed to be causing problems related to
    calling __init__ when using dataclasses. I don't quite recall exactly what
    was happening and was causing an issue, so it would be a good idea to try
    removing this metaclass and writing a test to make sure there was a problem
    to begin with, and also to make sure that adding back this class fixes it.
    """
    def __call__(cls, *args, **kwargs):
        # This is used to filter the arguments passed to the constructor
        # of the Setting and only keep the ones that are fields with init=True.
        init_fields: List[str] = [f.name for f in fields(cls) if f.init]
        extra_args: Dict[str, Any] = {}
        for k in list(kwargs.keys()):
            if k not in init_fields:
                extra_args[k] = kwargs.pop(k)
        if extra_args:
            logger.warning(UserWarning(
                f"Ignoring args {extra_args} when creating class {cls}."
            ))
        return super().__call__(*args, **kwargs)

@dataclass
class Setting(LightningDataModule, Serializable, Parseable, Generic[Loader], metaclass=SettingMeta):
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
    # Overwrite this in a subclass to customize which type of Results to create.
    results_class: ClassVar[Type[Results]] = Results

    # Transforms to be used. When no value is given for 
    # `[train/val/test]_transforms`, this value is used as a default.
    # TODO: Currently trying to find a way to specify the transforms from the
    # command-line, Therefore don't rely on that being done perfectly just yet.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels)
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
        
    _parent: ClassVar[Type["Setting"]] = None
    _children: ClassVar[List[Type["Setting"]]] = []
    _applicable_methods: ClassVar[Set[Type]] = set()
    
    def __post_init__(self,
                      obs_shape: Tuple[int, ...] = (),
                      action_shape: Tuple[int, ...] = (),
                      reward_shape: Tuple[int, ...] = ()):
        """ Initializes the fields of the setting that weren't set from the
        command-line.
        """
        logger.debug(f"__post_init__ of Setting")
        # Actually compose the list of Transforms or callables into a single transform.
        self.transforms: Compose = Compose(self.transforms)
        self.train_transforms: Compose = Compose(self.train_transforms or self.transforms)
        self.val_transforms: Compose = Compose(self.val_transforms or self.transforms)
        self.test_transforms: Compose = Compose(self.test_transforms or self.transforms)

        super().__init__(
            train_transforms=self.train_transforms,
            val_transforms=self.val_transforms,
            test_transforms=self.test_transforms,
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

        # This will be set when the setting gets configured
        self.config: Config = None

        # Wether the setup methods have been called yet or not.
        # TODO: Remove those, its just ugly and doesn't seem needed.
        self._prepared: bool = False
        self._configured: bool = False

    @abstractmethod
    def apply(self, method: "Method", config: Config) -> float:
        """ Applies the given Method on this experimental setting.
 
        Defines the training/evaluation procedure specific to this Setting.
        
        The training/evaluation loop can be defined however you want, as long as
        it respects the following constraints:
        
        1.  This method should always return a single float that indicates the
            "performance" of this method on this setting. We could assume that
            higher is better for now. 
        2. More importantly: You **have** to make sure that you do not break
            compatibility with more general methods targetting a parent setting!
            It should always be the case that all methods designed for any of
            this Setting's parents should also be applicable via polymorphism,
            i.e., anything that is defined to work on the class `Animal` should
            also work on the class `Cat`!
        3. While not enforced, it is strongly encourged that you define your
            training/evaluation routines at a pretty high level, so that Methods
            that get applied to your Setting can make use of pytorch-lightning's
            `Trainer` & `LightningDataModule` API to be neat and fast.
        """
    
    # def evaluate(self, method: "Method") -> ResultsType:
    #     """Tests the method and returns the Results.

    #     Overwrite this to customize testing for your experimental setting.

    #     Returns:
    #         ResultsType: A Results object for this particular setting.
    #     """
    #     from methods import Method
    #     method: Method
    #     trainer = method.trainer

    #     # Run the actual evaluation.
    #     assert trainer.datamodule is self
    #     test_outputs = trainer.test(
    #         datamodule=self,
    #         # verbose=False,
    #     )
    #     assert test_outputs, f"BUG: Pytorch lightning bug, Trainer.test() returned None!"
    #     test_loss: Loss = test_outputs[0]["loss_object"]

    #     model = method.model
    #     from methods.models import Model
    #     if isinstance(model, Model):
    #         hparams = model.hp
    #     else:
    #         assert False, f"TODO: Remove this ({model})."
    #         hparams = model.hparams
    #     return self.results_class(
    #         hparams=hparams,
    #         test_loss=test_loss,
    #     )

    @classmethod
    def main(cls, argv: Optional[Union[str, List[str]]]=None) -> ResultsType:
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

    def apply_all(self, argv: Union[str, List[str]]=None) -> Dict[Type["Method"], Results]:
        applicable_methods = self.get_all_applicable_methods()
        from methods import Method
        all_results: Dict[Type[Method], Results] = OrderedDict()
        for method_type in applicable_methods:
            method = method_type.from_args(argv)
            results = method.apply(self)
            all_results[method_type] = results
        logger.info(f"All results for setting of type {type(self)}:")
        logger.info({
            method.get_name(): (results.get_metric() if results else "crashed")
            for method, results in all_results.items()
        })
        return all_results

    @classmethod
    def get_all_applicable_methods(cls) -> List[Type["Method"]]:
        from methods import all_methods, Method
        return list(filter(lambda m: m.is_applicable(cls), all_methods))

    def __init_subclass__(cls, **kwargs):
        """ Called whenever a new subclass of `Setting` is declared. """
        assert is_dataclass(cls), f"Setting type {cls} isn't a dataclass!"
        logger.debug(f"Registering a new setting: {cls.get_name()}")

        # Exceptionally, create this new empty list that will hold all the
        # forthcoming subclasses of this particular new setting.
        cls.children = []
        # Inform all the nodes higher in the tree that they have a new subclass.
        parent = cls.__base__
        if issubclass(parent, Setting):
            parent: Type[Setting]
            assert cls not in parent._children
            parent._children.append(cls)
        # for t in cls.__bases__:
        #     if inspect.isclass(t) and issubclass(t, Setting):
        #         if cls not in t.sub_settings:
        #             t.sub_settings.append(cls)
        super().__init_subclass__(**kwargs)

    @classmethod
    def get_name(cls) -> str:
        """ Gets the name of this setting class. """
        # LightningDataModule has a `name` class attribute of `...`!
        if getattr(cls, "name", None) != Ellipsis:
            return cls.name
        else:
            name = camel_case(cls.__qualname__)
            return remove_suffix(name, "_setting")

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

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, value: List):
        if value:
            logger.warning(UserWarning(
                f"Setting the children attribute of class {type(self)} to a "
                f"non-empty list, are you sure you know what you're doing?"
            ))
        type(self)._children = value

    @property
    def all_children(self) -> Iterable[Type["Setting"]]:
        """Iterates over the inheritance tree, in-order.
        """
        # Yield the immediate children.
        for child in self._children:
            yield child
            # Yield from the children themselves.
            yield from child.all_children

    @property
    def parent(self) -> Optional[Type["Setting"]]:
        """Returns the first base class that is an instance of SettingMeta, else
        None
        """
        base_nodes = [
            base for base in type(self).__bases__ if isclass(base) and issubclass(base, Setting)
        ]
        return base_nodes[0] if base_nodes else None

    @property
    def parents(self) -> Iterable[Type["Setting"]]:
        """TODO: yields the lineage, from bottom to top. """
        parent = self.parent
        if parent:
            yield parent
            yield from parent.parents

