""" This module defines the `Setting` class, an ML "problem" to solve.

The `Setting` class is an abstract base class which should represent the most
general learning setting imaginable, i.e. with the fewest assumptions about the
data, the environment, the agent, etc.


The Setting class is currently loosely based on the `LightningDataModule` class
from pytorch-lightning, with the goal of having an `TraditionalSLSetting` node somewhere
in the tree, which would be totally interchangeable with existing datamodules
from pytorch-lightning.

The hope is that by staying close to that API, we can make it easier for people
to adopt the repo, and also, if possible, directly reuse existing models from
pytorch-lightning.

See: [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
See: [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html)

"""
import itertools
import sys
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Optional,
    TypeVar,
    Generic,
    ClassVar,
    Type,
    Dict,
    Any,
    List,
    Union,
    Iterable,
)

import gym
import numpy as np
import torch
from gym import spaces
from pytorch_lightning import LightningDataModule
from sequoia.common.config import Config, WandbConfig
from sequoia.common.metrics import Metrics
from sequoia.common.transforms import Compose, Transforms
from sequoia.settings.presets import setting_presets
from sequoia.utils import Parseable, get_logger, take
from simple_parsing import Serializable, field
from torch import Tensor

from .bases import Method, SettingABC
from .environment import Actions, Environment, Observations, Rewards
from .results import Results, ResultsType
from .setting_meta import SettingMeta

logger = get_logger(__file__)

SettingType = TypeVar("SettingType", bound="Setting")
EnvironmentType = TypeVar("EnvironmentType", bound=Environment)


@dataclass
class Setting(
    SettingABC,
    Parseable,
    Serializable,
    LightningDataModule,
    Generic[EnvironmentType],
):
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

    Abstract (required) methods:
    - **apply** Applies a given Method on this setting to produce Results.
    - **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    - **setup**  (things to do on every accelerator in distributed mode).
    - **train_dataloader** the training environment/dataloader.
    - **val_dataloader** the val environments/dataloader(s).
    - **test_dataloader** the test environments/dataloader(s).

    "Abstract"-ish (required) class attributes:
    - `Results`: The class of Results that are created when applying a Method on
      this setting.
    - `Observations`: The type of Observations that will be produced  in this
        setting.
    - `Actions`: The type of Actions that are expected from this setting.
    - `Rewards`: The type of Rewards that this setting will (potentially) return
      upon receiving an action from the method.
    """

    # ---------- Class Variables -------------
    # Fields in this block are class attributes. They don't create command-line
    # arguments.

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

    # Transforms to be applied to the observatons of the train/valid/test
    # environments.
    transforms: Optional[List[Transforms]] = None

    # Transforms to be applied to the training datasets.
    train_transforms: Optional[List[Transforms]] = None
    # Transforms to be applied to the validation datasets.
    val_transforms: Optional[List[Transforms]] = None
    # Transforms to be applied to the testing datasets.
    test_transforms: Optional[List[Transforms]] = None

    # Fraction of training data to use to create the validation set.
    # (Only applicable in Passive settings.)
    val_fraction: float = 0.2

    # TODO: Still not sure where exactly we should be adding the 'batch_size'
    # and 'num_workers' arguments. Adding it here for now with cmd=False, so
    # that they can be passed to the constructor of the Setting.
    batch_size: Optional[int] = field(default=None, cmd=False)
    num_workers: Optional[int] = field(default=None, cmd=False)

    # # TODO: Add support for semi-supervised training.
    # # Fraction of the dataset that is labeled.
    # labeled_data_fraction: int = 1.0
    # # Number of labeled examples.
    # n_labeled_examples: Optional[int] = None

    # Options related to Weights & Biases (wandb). Turned Off by default. Passing any of
    # its arguments will enable wandb.
    wandb: Optional[WandbConfig] = field(default=None, compare=False)
    
    def __post_init__(
        self,
        observation_space: gym.Space = None,
        action_space: gym.Space = None,
        reward_space: gym.Space = None,
    ):
        """ Initializes the fields of the setting that weren't set from the
        command-line.
        """
        logger.debug("__post_init__ of Setting")
        # BUG: simple-parsing sometimes parses a list with a single item, itself the
        # list of transforms. Not sure if this still happens.

        def is_list_of_list(v: Any) -> bool:
            return isinstance(v, list) and len(v) == 1 and isinstance(v[0], list)

        if is_list_of_list(self.train_transforms):
            self.train_transforms = self.train_transforms[0]
        if is_list_of_list(self.val_transforms):
            self.val_transforms = self.val_transforms[0]
        if is_list_of_list(self.test_transforms):
            self.test_transforms = self.test_transforms[0]

        # if all(
        #     t is None
        #     for t in [
        #         self.transforms,
        #         self.train_transforms,
        #         self.val_transforms,
        #         self.test_transforms,
        #     ]
        # ):
        #     # Use these two transforms by default if no transforms are passed at all.
        #     # TODO: Remove this after the competition perhaps.
        #     self.transforms = Compose([Transforms.to_tensor, Transforms.three_channels])

        # TODO: Should change this, so that these transform fields are only the
        # additional transforms compared to `self.transforms` (the 'base' transforms)
        # If the constructor is called with just the `transforms` argument, like this:
        # <SomeSetting>(dataset="bob", transforms=foo_transform)
        # Then we use this value as the default for the train, val and test transforms.
        if self.transforms and not any(
            [self.train_transforms, self.val_transforms, self.test_transforms]
        ):
            if not isinstance(self.transforms, list):
                self.transforms = Compose([self.transforms])
            self.train_transforms = self.transforms.copy()
            self.val_transforms = self.transforms.copy()
            self.test_transforms = self.transforms.copy()

        if self.train_transforms is not None and not isinstance(
            self.train_transforms, list
        ):
            self.train_transforms = [self.train_transforms]

        if self.val_transforms is not None and not isinstance(
            self.val_transforms, list
        ):
            self.val_transforms = [self.val_transforms]

        if self.test_transforms is not None and not isinstance(
            self.test_transforms, list
        ):
            self.test_transforms = [self.test_transforms]

        # Actually compose the list of Transforms or callables into a single transform.
        self.train_transforms: Compose = Compose(self.train_transforms or [])
        self.val_transforms: Compose = Compose(self.val_transforms or [])
        self.test_transforms: Compose = Compose(self.test_transforms or [])

        LightningDataModule.__init__(
            self,
            train_transforms=self.train_transforms,
            val_transforms=self.val_transforms,
            test_transforms=self.test_transforms,
        )

        self._observation_space = observation_space
        self._action_space = action_space
        self._reward_space = reward_space

        # TODO: It's a bit confusing to also have a `config` attribute on the
        # Setting. Might want to change this a bit.
        self.config: Config = None

        self.train_env: Environment = None  # type: ignore
        self.val_env: Environment = None  # type: ignore
        self.test_env: Environment = None  # type: ignore

    @abstractmethod
    def apply(self, method: Method, config: Config = None) -> "Setting.Results":
        # NOTE: The actual train/test loop should be defined in a more specific
        # setting. This is just here as an illustration of what that could look
        # like.
        assert False, "this is just here for illustration purposes. "

        method.fit(
            train_env=self.train_dataloader(), valid_env=self.val_dataloader(),
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
                batch_metrics = ...
                test_metrics.append(batch_metrics)
                if done:
                    break

        return self.Results(test_metrics=test_metrics)

    def get_metrics(self, actions: Actions, rewards: Rewards) -> Union[float, Metrics]:
        """ Calculate the "metric" from the model predictions (actions) and the true labels (rewards).

        In this example, we return a 'Metrics' object:
        - `ClassificationMetrics` for classification problems,
        - `RegressionMetrics` for regression problems.

        We use these objects because they are awesome (they basically simplify
        making plots, wandb logging, and serialization), but you can also just
        return floats if you want, no problem.

        TODO: This is duplicated from Incremental. Need to fix this.
        """
        from sequoia.common.metrics import get_metrics

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

        if isinstance(self.action_space, spaces.Discrete):
            batch_size = rewards.shape[0]
            actions = torch.as_tensor(actions)
            if len(actions.shape) == 1 or (
                actions.shape[-1] == 1 and self.action_space.n != 2
            ):
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
        logger.warning(
            f"Don't know what the image space is. "
            f"(self.observation_space={self.observation_space})"
        )
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
            raise RuntimeError(f"Value must be a `gym.Space` (got {value})")
        if not self._dims:
            if isinstance(value, spaces.Box):
                self.dims = value.shape
            elif isinstance(value, spaces.Tuple):
                self.dims = tuple(space.shape for space in value.spaces)
            elif isinstance(value, spaces.Dict) and "x" in value.spaces:
                self.dims = value.spaces["x"].shape
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
    def main(cls, argv: Optional[Union[str, List[str]]] = None) -> Results:
        from sequoia.main import Experiment

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

    def apply_all(
        self, argv: Union[str, List[str]] = None
    ) -> Dict[Type["Method"], Results]:
        applicable_methods = self.get_applicable_methods()
        from sequoia.methods import Method

        all_results: Dict[Type[Method], Results] = {}
        config = Config.from_args(argv)
        for method_type in applicable_methods:
            method = method_type.from_args(argv)
            results = self.apply(method, config)
            all_results[method_type] = results
        logger.info(f"All results for setting of type {type(self)}:")
        logger.info(
            {
                method.get_name(): (results.get_metric() if results else "crashed")
                for method, results in all_results.items()
            }
        )
        return all_results

    def _check_environments(self):
        """ Do a quick check to make sure that interacting with the envs/dataloaders
        works correctly.
        """
        # Check that the env's spaces are batched versions of the settings'.
        from gym.vector.utils import batch_space
        from sequoia.settings.sl import PassiveEnvironment

        batch_size = self.batch_size
        for loader_method in [
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        ]:
            print(f"\n\nChecking loader method {loader_method.__name__}\n\n")
            env = loader_method(batch_size=batch_size)

            batch_size = env.batch_size

            # We could compare the spaces directly, but that's a bit messy, and
            # would be depends on the type of spaces for each. Instead, we could
            # check samples from such spaces on how the spaces are batched.
            if batch_size:
                expected_observation_space = batch_space(
                    self.observation_space, n=batch_size
                )
                expected_action_space = batch_space(self.action_space, n=batch_size)
                expected_reward_space = batch_space(self.reward_space, n=batch_size)
            else:
                expected_observation_space = self.observation_space
                expected_action_space = self.action_space
                expected_reward_space = self.reward_space

            # TODO: Batching the 'Sparse' makes it really ugly, so just
            # comparing the 'image' portion of the space for now.
            assert (
                env.observation_space[0].shape == expected_observation_space[0].shape
            ), (env.observation_space[0], expected_observation_space[0])

            assert env.action_space == expected_action_space, (
                env.action_space,
                expected_action_space,
            )
            assert env.reward_space == expected_reward_space, (
                env.reward_space,
                expected_reward_space,
            )

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

            env.close()

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
            raise RuntimeError(
                f"Don't know how to find the image space in the "
                f"env's obs space ({env.observation_space})."
            )
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
            rewards = rewards.y
        if isinstance(rewards, Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(rewards, np.ndarray):
            rewards = rewards
        if isinstance(rewards, (int, float)):
            rewards = np.asarray(rewards)
        assert rewards in env.reward_space, (rewards, env.reward_space)

    # Just to make type hinters stop throwing errors when using the constructor
    # to create a Setting.
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def load_benchmark(
        cls: Type[SettingType], benchmark: Union[str, Path]
    ) -> SettingType:
        """ Load the given "benchmark" (pre-configured Setting) of this type.

        Parameters
        ----------
        cls : Type[SettingType]
            Type of Setting to create.
        benchmark : Union[str, Path]
            Either the name of a benchmark (e.g. "cartpole_state", "monsterkong", etc.)
            or a path to a json/yaml file.

        Returns
        -------
        SettingType
            Setting of type `cls`, appropriately populated according to the chosen
            benchmark.

        Raises
        ------
        RuntimeError
            If `benchmark` isn't an existing file or a known preset.
        RuntimeError
            If any command-line arguments are present in sys.argv which would be ignored
            when creating this setting.
        """
        # If the provided benchmark isn't a path, try to get the value from
        # the `setting_presets` dict. If it isn't in the dict, raise an
        # error.
        if not Path(benchmark).is_file():
            if benchmark in setting_presets:
                benchmark = setting_presets[benchmark]
            else:
                raise RuntimeError(
                    f"Could not find benchmark '{benchmark}': it "
                    f"is neither a path to a file or a key of the "
                    f"`setting_presets` dictionary. \n"
                    f"(Available presets: {setting_presets}) "
                )
        # Creating an experiment for the given setting, loaded from the
        # config file.
        # TODO: IDEA: Do the same thing for loading the Method?
        logger.info(
            f"Will load the options for setting {cls} from the file "
            f"at path {benchmark}."
        )

        # Raise an error if any of the args in sys.argv would have been used
        # up by the Setting, just to prevent any ambiguities.
        _, unused_args = cls.from_known_args()
        consumed_args = list(set(sys.argv[1:]) - set(unused_args))
        if consumed_args:
            # TODO: This could also be trigerred if there were arguments
            # in the method with the same name as some from the Setting.
            raise RuntimeError(
                f"Cannot pass command-line arguments for the Setting when "
                f"loading a benchmark, since these arguments whould have been "
                f"ignored when creating the setting of type {cls} "
                f"anyway: {consumed_args}"
            )

        drop_extras = False
        # Actually load the setting from the file.
        setting = cls.load(path=benchmark, drop_extra_fields=drop_extras)
        return setting
