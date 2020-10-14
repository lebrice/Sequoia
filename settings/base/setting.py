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
import itertools
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

import gym
import torch
import numpy as np
from gym import spaces
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.datamodule import _DataModuleWrapper
from simple_parsing import (ArgumentParser, Serializable, list_field,
                            mutable_field, subparsers, field)
from torch.utils.data import DataLoader

from common.config import Config
from common.loss import Loss
from common.metrics import Metrics
from common.transforms import Compose, Transforms, Transform, SplitBatch
from utils import Parseable, camel_case, dict_union, get_logger, remove_suffix, take

from .results import Results
from .setting_meta import SettingMeta
from .environment import Environment, Observations, Actions, Rewards

logger = get_logger(__file__)

EnvironmentType = TypeVar("EnvironmentType", bound=Environment)
SettingType = TypeVar("SettingType", bound="Setting")

from ..setting_base import SettingABC
from ..method_base import MethodABC

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
    # x_shape: Tuple[int, ...] = field(init=False)
    # action_shape: Tuple[int, ...] = field(init=False)
    # reward_shape: Tuple[int, ...] = field(init=False)
    
    def __post_init__(self,
                      observation_space: gym.Space = None,
                      action_space: gym.Space = None,
                      reward_space: gym.Space = None):
        """ Initializes the fields of the setting that weren't set from the
        command-line.
        """
        logger.debug(f"__post_init__ of Setting")
        
        # Actually compose the list of Transforms or callables into a single transform.
        self.transforms: Compose = Compose(self.transforms)
        self.train_transforms: Compose = Compose(self.train_transforms or self.transforms)
        self.val_transforms: Compose = Compose(self.val_transforms or self.transforms)
        self.test_transforms: Compose = Compose(self.test_transforms or self.transforms)

        LightningDataModule.__init__(self,
            train_transforms=self.train_transforms,
            val_transforms=self.val_transforms,
            test_transforms=self.test_transforms,
        )
        
        self._observation_space = observation_space
        self._action_space = action_space
        self._reward_space = reward_space

        
        # # Transform that will split batches into Observations and Rewards.
        # self.split_batch_transform = SplitBatch(
        #     observation_type=self.Observations,
        #     reward_type=self.Rewards
        # )
        logger.debug(f"Transforms: {self.transforms}")

        # TODO: Testing out an idea: letting the transforms tell us how
        # they change the shape of the observations.
        # if x_shape and self.transforms:
        #     logger.debug(f"x shape before transforms: {x_shape}")
        #     x_shape: Tuple[int, ...] = self.transforms.shape_change(x_shape)
        #     logger.debug(f"x shape after transforms: {x_shape}")
        # self.observation_space = x_shape

        self.dataloader_kwargs: Dict[str, Any] = {}

        # TODO: We have to set the 'dims' property from LightningDataModule so
        # that models know the input dimensions.
        # This should probably be set on `self` inside of `apply` call.
        # TODO: It's a bit confusing to also have a `config` attribute on the
        # Setting. Might want to change this a bit.
        self.config: Config = None

    @abstractmethod
    def apply(self, method: MethodABC, config: Config) -> "Setting.Results":
        assert False, "this should never be called."
        method.fit(
            train_env=self.train_dataloader(),
            valid_env=self.val_dataloader(),
        )

        # Test loop:
        test_metrics: List[Metrics] = []
        test_env = self.test_dataloader()

        # Get initial observations.
        observations = test_env.reset()
        # get actions for a batch of observations.
        actions = method.get_actions(observations, test_env.action_space)

        for i in itertools.count():
            observations, rewards, done, info = test_env.step(actions)
            # Get the predictions/actions:
            actions = method.get_actions(observations, test_env.action_space)
            # Calculate the 'metrics' (TODO: This should be done be in the env!)
            batch_metrics = self.get_metrics(actions=actions, rewards=rewards)
            test_metrics.append(batch_metrics)

            if done:
                break

        return self.Results(test_metrics=test_metrics)

    def get_metrics(self,
                    actions: Actions,
                    rewards: Rewards) -> Union[float, Metrics]:
        """ Calculate the "metric" from the model predictions (actions) and the true labels (rewards).
        
        In this example, we return a 'Metrics' object:
        - `ClassificationMetrics` for classification problems,
        - `RegressionMetrics` for regression problems.
        
        We use these objects because they are awesome (they basically simplify
        making plots, wandb logging, and serialization), but you can also just
        return floats if you want, no problem.
        
        TODO: This is duplicated from Incremental. Need to fix this.
        """
        from common.metrics import get_metrics
        # In this particular setting, we only use the y_pred from actions and
        # the y from the rewards.
        if isinstance(actions, Actions):
            actions = torch.as_tensor(actions.y_pred)
        if isinstance(rewards, Rewards):
            rewards = torch.as_tensor(rewards.y)
        # TODO: At the moment there's this problem, ClassificationMetrics wants
        # to create a confusion matrix, which requires 'logits' (so it knows how
        # many classes.
        if isinstance(self.action_space, spaces.Discrete):
            batch_size = rewards.shape[0]
            actions = torch.as_tensor(actions)
            if len(actions.shape) == 1 or (actions.shape[-1] == 1 and self.action_space.n != 2):
                fake_logits = torch.zeros([batch_size, self.action_space.n], dtype=int)
                # FIXME: There must be a smarter way to do this indexing.
                for i, action in enumerate(actions):
                    fake_logits[i, action] = 1
                actions = fake_logits

        return get_metrics(y_pred=actions, y=rewards)

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: gym.Space) -> None:
        """Sets a the observation space.
        
        NOTE: This also changes the value of the `dims` attribute and the result
        of the `size()` method from LightningDataModule.
        """
        if not isinstance(value, gym.Space):
            raise RuntimeError("Value must be a `gym.Space`.")
        if not self._dims:
            if isinstance(value, spaces.Box):
                self.dims = value.shape
            elif isinstance(value, spaces.Tuple):
                self.dims = tuple(space.shape for space in value.spaces)
            else:
                raise NotImplementedError(
                    f"Don't know how to set the 'dims' attribute using "
                    f"observation space {value}"
                )
        self._observation_space = value

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @action_space.setter
    def action_space(self, value: gym.Space) -> None:
        self._action_space = value

    @property
    def reward_space(self) -> gym.Space:
        return self._reward_space

    @reward_space.setter
    def reward_space(self, value: gym.Space) -> None:
        self._reward_space = value
            
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

    def configure(self, method: MethodABC):
        """ Setup the data_dir and the dataloader kwargs using properties of the
        Method or of self.Config.

        Parameters
        ----------
        method : MethodABC
            The Method that is being applied on this setting.
        config : Config
            [description]
        """
        assert self.config is not None
        config = self.config
        # Get the arguments that will be used to create the dataloaders.
        
        # TODO: Should the data_dir be in the Setting, or the Config?
        self.data_dir = config.data_dir
        
        # Create the dataloader kwargs, if needed.
        if not self.dataloader_kwargs:
            batch_size = 32
            if hasattr(method, "batch_size"):
                batch_size = method.batch_size
            elif hasattr(method, "model") and hasattr(method.model, "batch_size"):
                batch_size = method.model.batch_size
            elif hasattr(config, "batch_size"):
                batch_size = config.batch_size

            dataloader_kwargs = dict(
                batch_size=batch_size,
                num_workers=config.num_workers,
                shuffle=False,
            )
        # Save the dataloader kwargs in `self` so that calling `train_dataloader()`
        # from outside with no arguments (i.e. when fitting the model with self
        # as the datamodule) will use the same args as passing the dataloaders
        # manually.
        self.dataloader_kwargs = dataloader_kwargs
        logger.debug(f"Dataloader kwargs: {dataloader_kwargs}")

        # Debugging: Run a quick check to see that what is returned by the
        # dataloaders is of the right type and shape etc.
        self._check_dataloaders_give_correct_types()

    def _check_dataloaders_give_correct_types(self):
        """ Do a quick check to make sure that the dataloaders give back the
        right observations / reward types.
        """
        for loader_method in [self.train_dataloader, self.val_dataloader, self.test_dataloader]:
            env = loader_method()
            from settings.passive import PassiveEnvironment
            from settings.active import ActiveEnvironment
            if isinstance(env, PassiveEnvironment):
                print(f"{env} is a PassiveEnvironment!")
            else:
                print(f"{env} is an ActiveEnvironment!")

            for batch in take(env, 5):
                if isinstance(env, PassiveEnvironment):
                    observations, *rewards = batch
                else:
                    observations, rewards = batch, None
                try:
                    assert isinstance(observations, self.Observations), type(observations)
                    observations: Observations
                    batch_size = observations.batch_size
                    rewards: Optional[Rewards] = rewards[0] if rewards else None
                    if rewards is not None:
                        assert isinstance(rewards, self.Rewards), type(rewards)
                    # TODO: If we add gym spaces to all environments, then check
                    # that the observations are in the observation space, sample
                    # a random action from the action space, check that it is
                    # contained within that space, and then get a reward by
                    # sending it to the dataloader. Check that the reward
                    # received is in the reward space.
                    actions = self.action_space.sample()
                    assert actions.shape[0] == batch_size
                    actions = self.Actions(torch.as_tensor(actions))
                    rewards = env.send(actions)
                    assert isinstance(rewards, self.Rewards), type(rewards)
                except Exception as e:
                    logger.error(f"There's a problem with the method {loader_method} (env {env})")
                    raise e
    