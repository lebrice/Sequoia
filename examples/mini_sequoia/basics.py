""" This file contains a minimal reproduction of the main components of Sequoia.

The main sections of this file describe (in order):

1. Introduction: `Space`, `Observation`, `Actions`, `Rewards`, and `Environment`;
2. The "Setting": represents a research problem from the litterature;
2. The "Method": representing a solution to such a research problem.

If you want to see an actual (runnable) example for a Method using PyTorch-Lightning,
take a look at
https://github.com/lebrice/Sequoia/blob/master/examples/basic/pl_example.py
"""
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import gym
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

S = TypeVar("S")

# --------------------- utility classes (used later) -------------------


class Space(gym.Space, Generic[S]):
    """Generic version of gym.Space."""

    dtype: Type[S]

    def sample(self) -> S:
        return super().sample()

    def contains(self, s: Union[S, Any]) -> bool:
        return super().contains(s)


@dataclass(frozen=True)
class Batch:
    """Base class for a Dataclass with tensor fields, with some convenience methods."""

    def _map(self, func: Callable):
        """returns an object of the same type by applying `func` to all items."""
        return dataclasses.replace(
            self,
            **{f.name: func(getattr(self, f.name)) for f in dataclasses.fields(self)},
        )

    def to(self, *args, **kwargs):
        def _to(v: Any) -> Any:
            if hasattr(v, "to") and callable(v.to):
                return v.to(*args, **kwargs)
            return v

        return self._map(_to)


"""
--------------------------- "Setting" API ---------------------------

Here we introduce a few concepts which are used throughout Sequoia: observations,
actions, rewards, environment, and, most importantly: Setting.

NOTE: By the way, it's not important for this example since we're only dealing with SL,
but just FYI:

Environment :== DataLoader + gym.Env

The DataLoaders (a.k.a. "Environments") returned by the `[train/val/test]_dataloader`
methods of a `Setting` in Sequoia (shown below) inherit from both `DataLoader`
as well as `gym.Env`!
This makes it easier to create Methods applicable to both RL and SL Settings!
"""


@dataclass(frozen=True)
class Observations(Batch):
    """The observations/samples produced by an Environment."""

    # All observations must at least have an `x` field, which may represent different things
    # depending on the setting, but is usually just a batch of samples (tensors).
    x: Tensor


@dataclass(frozen=True)
class Actions(Batch):
    """The actions/predictions that are sent to an Environment."""

    # All Actions must at least have a `y_pred` field, which may represent different things
    # depending on the setting, but are usually just a batch of predictions (tensors).
    y_pred: Any


@dataclass(frozen=True)
class Rewards(Batch):
    """The rewards/labels returned by an Environment."""

    # All `Rewards` subclasses must at least have a `y` field, which may represent different
    # things depending on the setting, but are a batch of labels/rewards from the environment
    # (a Tensor).
    y: Any


# Type variables, just for fun. The Environment and Model classes are generics of these.
ObservationType = TypeVar("ObservationType", bound=Observations)
ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)


class Environment(
    DataLoader, gym.Env, Generic[ObservationType, ActionType, RewardType], ABC
):
    """Pseudocode / type hints for the Environment: a fusion of DataLoader + gym.Env.

    DataLoaders in Sequoia  more on this later) return structured
    objects (dataclasses) rather than just Tensors or tuples of tensors.

    This makes it possible to reuse Methods on different Settings via Polymorphism,
    which is a big deal for reproducible research - but this is a conversation for
    another day.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        observation_space: Space[ObservationType],
        action_space: Space[ActionType],
        reward_space: Space[RewardType],
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        gym.Env.__init__(self)
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_space = reward_space
        # temporary holder for the rewards.
        self._rewards: RewardType

    @abstractmethod
    def __iter__(self) -> Iterator:  # type: ignore
        return super().__iter__()

    def send(self, action: ActionType) -> RewardType:
        return self._rewards

    # NOTE: Not implementing these gym-style methods in this example just to keep things simple,
    # since we're only using the DataLoader API and don't call `step` or `reset` below.
    # Although you can probably imagine how `reset` and `step` might work: They simply advance the
    # dataloader iterator one step, and return the contents. The "actions" in SL don't have an
    # influence on the iterator's next observation, while in RL they do.

    def reset(self) -> ObservationType:
        ...

    def step(
        self, actions: ActionType
    ) -> Tuple[ObservationType, RewardType, bool, dict]:
        ...

    def render(self) -> None:
        ...


class TestEnvironment(Environment[ObservationType, ActionType, RewardType], ABC):
    """Special environment used for testing.

    It doesn't yield the `Rewards` until an `Action` is sent to its `send` method.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        observation_space: Space[ObservationType],
        action_space: Space[ActionType],
        reward_space: Space[RewardType],
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size,
            observation_space,
            action_space,
            reward_space,
            num_workers=num_workers,
            **kwargs,
        )
        self._metrics: List[float] = []

    @abstractmethod
    def __iter__(self) -> Iterator:  # type: ignore
        return super().__iter__()

    def send(self, actions: ActionType) -> RewardType:
        self._metrics.append(self.get_metric(actions=actions, rewards=self._rewards))
        return self._rewards

    @abstractmethod
    def get_metric(self, actions: ActionType, rewards: RewardType) -> float:
        pass

    def get_performance(self) -> float:
        """Returns the average performance (in this case the average accuracy)"""
        if not self._metrics:
            return 0.0
        return sum(self._metrics) / len(self._metrics)


class Setting(LightningDataModule, ABC):
    """Example of a "Setting": a research problem.

    Methods can be applied onto Settings to produce Results.

    The `Setting` class inherits from `LightningDataModule`.
    It adds a single method to that class:
    - `<Setting>.apply(method: Method) -> <Setting>.Results`
        Runs the entire train/test loop.

    Added properties (which aren't shown in this example):
    - `observation_space: gym.Space`
    - `action_space: gym.Space`
    - `reward_space: gym.Space`
    """

    class Results(ABC):
        """Object containing the results of the experiment in which a Method is applied
        to this type of Setting.

        Results must define an 'objective' which is a single float that represents the
        overall performance, can be used to compare different experiments.
        """

        @property
        @abstractmethod
        def objective(self) -> float:
            """
            A single float that represents the overall performance of the Method on the
            Setting. Can be used to compare different experiments.
            """

    # Save the types defined above as attributes on the Setting class itself.
    # This is useful when defining a new Setting in the next step, because its observations,
    # actions, rewards, environments, etc. will inherit from these.
    # This guarantees that a Method can be applied on any Subclass of their target Setting (which is
    # the main selling point of Sequoia, more on this later).
    Environment: ClassVar[Type[Environment]] = Environment
    TestEnvironment: ClassVar[Type[TestEnvironment]] = TestEnvironment
    Observations: ClassVar[Type[Observations]] = Observations
    Actions: ClassVar[Type[Actions]] = Actions
    Rewards: ClassVar[Type[Rewards]] = Rewards

    # Settings define an observation/action/reward spaces
    observation_space: Space[Observations]
    action_space: Space[Actions]
    reward_space: Space[Rewards]

    @abstractmethod
    def apply(self, method: "Method") -> "Setting.Results":
        """Apply a Method onto this Setting, producing Results."""

    @abstractmethod
    def train_dataloader(self) -> Environment:
        """Returns a training environment/dataloader."""

    @abstractmethod
    def val_dataloader(self) -> Environment:
        """Returns a validation environment/dataloader."""

    @abstractmethod
    def test_dataloader(self) -> TestEnvironment:
        """Returns the testing environment/dataloader."""
        # NOTE: In Sequoia: The test dataloader is a bit different than the train/val
        # dataloaders: It doesn't give the "rewards" (e.g. image labels `y`) until after it
        # receives the "action" (e.g. predicted class `y_pred`) in its `send` method.

    def prepare_data(self) -> None:
        """Download the data required to create the datasets/environments."""

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the datasets/environments."""


"""
----------------------------- "Method" API: ---------------------------------------

Here we introduce the Method API:
"""

SettingType = TypeVar("SettingType", bound=Setting)


class Method(ABC, Generic[SettingType]):
    """Template/pseudo-code of a "Method": A solution to a Setting (research problem).

    This is meant as an illustration of the responsibilities of a `Method` in Sequoia.
    The required methods are
    - `configure(setting: Setting) -> None`
        Configure the method before it is applied onto the Setting;
    - `fit(train_env: DataLoader, valid_env: DataLoader) -> None`
        Train the Method using the provided training environment, and validate the method
    - as well as one of:
        - `get_actions(observations: Observations) -> Actions`
            Return a single `Action` (prediction) for the corresponding observation/sample.
        - `test(test_env: TestEnvironment) -> None`
            Iterate over the test environment yourself (e.g. using something like
            `self.trainer.test(test_env)`.

    It is recommended for Methods to use the `pl.LightningModule` and `pl.Trainer`
    classes, since they make code a lot easier to read and enable lots of cool features.
    NOTE: (You don't *have* to use PyTorch-Lightning if you don't want to.)
    """

    target_setting: ClassVar[Type[SettingType]]

    training: bool

    @abstractmethod
    def configure(self, setting: SettingType) -> None:
        """Called by the Setting to give the Method the opportunity to customize itself
        before training begins.
        """

    @abstractmethod
    def fit(
        self, train_env: "SettingType.Environment", valid_env: "SettingType.Environment"
    ) -> None:  # type: ignore
        """Called by the Setting to allow the Method to train and validate itself."""

    def test(self, test_env: "SettingType.TestEnvironment") -> None:  # type: ignore
        """Iterate over the test environment ourselves.
        The test environment will report back our performance to the Setting.
        """
        raise NotImplementedError

    def get_actions(
        self, observations: "SettingType.Observations", action_space: Space["SettingType.Actions"]  # type: ignore
    ) -> "SettingType.Actions":  # type: ignore
        raise NotImplementedError

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Called by some Settings when there are multiple tasks and we encounter a
        task boundary.

        If the Setting doesn't provide task identity during the current phase, then
        `task_id` will be `None`, otherwise it is the index of the current task.
        """
