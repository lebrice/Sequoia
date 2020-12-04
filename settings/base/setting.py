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
                            mutable_field, subparsers, field, choice)
from torch import Tensor
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

from  .bases import SettingABC, Method


@dataclass
class Setting(SettingABC,
              Generic[EnvironmentType],
              Serializable,
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
    
    available_datasets: ClassVar[Dict[str, Any]] = {}
    
    ##
    ##   -------------
    
    # Transforms to be applied to the training datasets.
    train_transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.three_channels)
    # Transforms to be applied to the validation datasets. 
    val_transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.three_channels)
    # Transforms to be applied to the testing datasets.
    test_transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.three_channels)

    # Fraction of training data to use to create the validation set.
    # (Only applicable in Passive settings.)
    val_fraction: float = 0.2

    # TODO: Add support for semi-supervised training.
    # Fraction of the dataset that is labeled.
    labeled_data_fraction: int = 1.0
    # Number of labeled examples.
    n_labeled_examples: Optional[int] = None

    def __post_init__(self,
                      observation_space: gym.Space = None,
                      action_space: gym.Space = None,
                      reward_space: gym.Space = None):
        """ Initializes the fields of the setting that weren't set from the
        command-line.
        """
        logger.debug(f"__post_init__ of Setting")
        
        # Actually compose the list of Transforms or callables into a single transform.
        self.train_transforms: Compose = Compose(self.train_transforms)
        self.val_transforms: Compose = Compose(self.val_transforms)
        self.test_transforms: Compose = Compose(self.test_transforms)

        LightningDataModule.__init__(self,
            train_transforms=self.train_transforms,
            val_transforms=self.val_transforms,
            test_transforms=self.test_transforms,
        )
        
        self._observation_space = observation_space
        self._action_space = action_space
        self._reward_space = reward_space

        # TODO: Testing out an idea: letting the transforms tell us how
        # they change the shape of the observations.
        # if x_shape and self.transforms:
        #     logger.debug(f"x shape before transforms: {x_shape}")
        #     x_shape: Tuple[int, ...] = self.transforms.shape_change(x_shape)
        #     logger.debug(f"x shape after transforms: {x_shape}")
        # self.observation_space = x_shape

        self.batch_size: Optional[int] = None
        self.num_workers: Optional[int] = None
        # TODO: We have to set the 'dims' property from LightningDataModule so
        # that models know the input dimensions.
        # This should probably be set on `self` inside of `apply` call.
        # TODO: It's a bit confusing to also have a `config` attribute on the
        # Setting. Might want to change this a bit.
        self.config: Config = None

    @abstractmethod
    def apply(self, method: Method, config: Config = None) -> "Setting.Results":
        # NOTE: The actual train/test loop should be defined in a more specific
        # setting. This is just here as an illustration of what that could look
        # like. 
        assert False, "this is just here for illustration purposes. "
        
        method.fit(
            train_env=self.train_dataloader(),
            valid_env=self.val_dataloader(),
        )

        # Test loop:
        test_env = self.test_dataloader()
        test_metrics = []
        # Number of episodes to test on:
        n_test_episodes = 1

        # Perform a set number of episodes in the test environment.
        for episode in range(n_test_episodes):
            # Get initial observations.
            observations = test_env.reset()
            
            for i in itertools.count():
                # Get the predictions/actions for a batch of observations.
                actions = method.get_actions(observations, test_env.action_space)
                observations, rewards, done, info = test_env.step(actions)
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
        if isinstance(actions, Tensor):
            actions = actions.cpu().numpy()
        if isinstance(rewards, Tensor):
            rewards = rewards.cpu().numpy()
        
        # assert actions in self.action_space, f"Invalid actions {actions} (space = {self.action_space})"
        # assert rewards in self.reward_space, f"Invalid rewards? {rewards} (space = {self.reward_space})"
        
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
    def image_space(self) -> Optional[gym.Space]:
        if isinstance(self.observation_space, spaces.Box):
            return self.observation_space
        if isinstance(self.observation_space, spaces.Tuple):
            assert isinstance(self.observation_space[0], spaces.Box)
            return self.observation_space[0]
        if isinstance(self.observation_space, spaces.Dict):
            return self.observation_space.spaces["x"]
        logger.warning(f"Don't know what the image space is. "
                       f"(self.observation_space={self.observation_space})")
        return None

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
    def get_available_datasets(cls) -> Iterable[str]:
        """ Returns an iterable of strings which represent the names of datasets. """
        return cls.available_datasets

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
    def get_path_to_source_file(cls: Type) -> Path:
        from utils.utils import get_path_to_source_file
        return get_path_to_source_file(cls)

    def configure(self, method: Method):
        """ Configure the setting before the method is applied to it.
        
        TODO: This is basically just here so we can figure out the batch size
        to use and the directory where the data should be downloaded, which are
        properties on the Config object (which atm is not in the setting, but on
        either the Method or the Experiment.). Need to clean this up.
        
        Parameters
        ----------
        method : Method
            The Method that is being applied on this setting.
        config : Config
            [description]
        """
        # TODO: Remove this, move it to `prepare_data`.
        assert self.config is not None
        # TODO: Should the data_dir be in the Setting, or the Config?
        self.data_dir = self.config.data_dir
        # Create the dataloader kwargs, if needed.
        self.batch_size = self.batch_size or getattr(self.config, "batch_size", None)
        self.num_workers = self.num_workers or self.config.num_workers

        # Debugging: Run a quick check to see that what is returned by the
        # dataloaders is of the right type and shape etc.
        if self.config.debug:
            self._check_environments()


    def _check_environments(self):
        """ Do a quick check to make sure that interacting with the envs/dataloaders
        works correctly.
        """
        from settings.passive import PassiveEnvironment
        from settings.active import ActiveEnvironment
        
        # Check that the env's spaces are batched versions of the settings'.
        from gym.vector.utils import batch_space

        batch_size = self.batch_size
        for loader_method in [self.train_dataloader, self.val_dataloader, self.test_dataloader]:
            print(f"\n\nChecking loader method {loader_method.__name__}\n\n")
            env = loader_method(batch_size=batch_size)
            batch_size = env.batch_size

            # We could compare the spaces directly, but that's a bit messy, and
            # would be depends on the type of spaces for each. Instead, we could
            # check samples from such spaces on how the spaces are batched. 
            if batch_size:
                expected_observation_space = batch_space(self.observation_space, n=batch_size)
                expected_action_space = batch_space(self.action_space, n=batch_size)
                expected_reward_space = batch_space(self.reward_space, n=batch_size)
            else:
                expected_observation_space = self.observation_space
                expected_action_space = self.action_space
                expected_reward_space = self.reward_space
            
            # TODO: Batching the 'Sparse' makes it really ugly, so just
            # comparing the 'image' portion of the space for now.
            assert env.observation_space[0].shape == expected_observation_space[0].shape, (env.observation_space[0], expected_observation_space[0])
            # assert env.observation_space[0] == expected_observation_space[0], (env.observation_space[0], expected_observation_space[0])
            # assert env.observation_space[1] == expected_observation_space[1], (
            #     f"env obs space: {env.observation_space[1]}, \n"
            #     f"expected obs space: {expected_observation_space[1]}"
            # )

            assert env.action_space == expected_action_space, (env.action_space, expected_action_space)
            assert env.reward_space == expected_reward_space, (env.reward_space, expected_reward_space)

            # Check that the 'gym API' interaction is working correctly.
            reset_obs: Observations = env.reset()
            self._check_observations(env, reset_obs)

            for i in range(5):
                actions = env.action_space.sample()
                self._check_actions(env, actions)
                step_observations, step_rewards, done, info = env.step(actions)
                self._check_observations(env, step_observations)
                self._check_rewards(env, step_rewards)
                if batch_size:
                    assert not any(done)
                else:
                    assert not done
                # assert not (done if isinstance(done, bool) else any(done))

            for batch in take(env, 5):
                observations: Observations
                rewards: Optional[Rewards]
                
                if isinstance(env, PassiveEnvironment):
                    observations, rewards = batch
                else:
                    # in RL atm, the 'dataset' gives back only the observations.
                    # Coul
                    observations, rewards = batch, None

                self._check_observations(env, observations)
                if rewards is not None:
                    self._check_rewards(env, rewards)
                
                if batch_size:
                    actions = tuple(
                        self.action_space.sample() for _ in range(batch_size)
                    )
                else:
                    actions = self.action_space.sample()
                # actions = self.Actions(torch.as_tensor(actions))
                rewards = env.send(actions)
                self._check_rewards(env, rewards)
    
    def _check_observations(self, env: Environment, observations: Any):
        """ Check that the given observation makes sense for the given environment.
        
        TODO: This should probably not be in this file here. It's more used for
        testing than anything else.
        """
        assert isinstance(observations, self.Observations), observations
        images = observations.x
        assert isinstance(images, (torch.Tensor, np.ndarray))
        if isinstance(images, Tensor):
            images = images.cpu().numpy()
        
        # Find the 'image' space:
        if isinstance(env.observation_space, spaces.Box):
            image_space = env.observation_space
        elif isinstance(env.observation_space, spaces.Tuple):
            image_space = env.observation_space[0]
        else:
            raise RuntimeError(f"Don't know how to find the image space in the "
                               f"env's obs space ({env.observation_space}).")
        assert images in image_space
    
    def _check_actions(self, env: Environment, actions: Any):
        if isinstance(actions, Actions):
            assert isinstance(actions, self.Actions)
            actions = actions.y_pred.cpu().numpy()
        elif isinstance(actions, Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, np.ndarray):
            actions = actions
        assert actions in env.action_space
    
    def _check_rewards(self, env: Environment, rewards: Any):
        if isinstance(rewards, Rewards):
            assert isinstance(rewards, self.Rewards)
            rewards = rewards.y.cpu().numpy()
        elif isinstance(rewards, Tensor):
            rewards = rewards.cpu().numpy()
        elif isinstance(rewards, np.ndarray):
            rewards = rewards
        assert rewards in env.reward_space

    # Just to make type hinters stop throwing errors when using the constructor
    # to create a Setting.
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)
