from __future__ import annotations

""" This example builds on top of `basics.py` and builds a concrete, minimal example of a Setting
and a Method.

NOTE: We use the `continuum` package (https://github.com/Continvvm/continuum) to create and split
the datasets for each task.
"""


from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from collections import deque
from typing import (
    Any,
    ClassVar,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch

from continuum import ClassIncremental, datasets
from continuum.tasks import split_train_val
from gym import spaces
from pytorch_lightning import Trainer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from examples.mini_sequoia.basics import (
    Method,
    S,
    Setting,
    Space,
)

Stage = Literal["fit", "validate", "test", "predict"]


class TypedDictSpace(spaces.Dict, Space[S]):
    """Typed version of spaces.Dict, which adds a `dtype` argument.
    (This is used with dataclasses below)
    """

    def __init__(
        self,
        spaces: Optional[Mapping[str, Space]] = None,
        dtype: Type[S] = dict,
        **spaces_kwargs: Space,
    ) -> None:
        super().__init__(spaces=spaces, **spaces_kwargs)
        self.dtype = dtype

    def sample(self) -> S:
        d = super().sample()
        return self.dtype(**d)

    def contains(self, v: Union[S, Any]) -> bool:
        if is_dataclass(v):
            v = asdict(v)
        return super().contains(v)

    def __getattr__(self, attr: str):
        if attr != "spaces" and attr in self.spaces:
            return self.spaces[attr]
        raise AttributeError(attr)


@dataclass(frozen=True)
class Observations(Setting.Observations):
    """The observations/samples produced by an Environment in Task-Incremental SL."""

    # In this example, `x` is a Tensor containing a batch of images.
    x: Tensor
    # In this example, `task_labels` is the task label associated with each image. Some settings
    # might provide it, while others might not.
    task_labels: Optional[Tensor] = None


@dataclass(frozen=True)
class Actions(Setting.Actions):
    """The actions/predictions that are sent to an Environment in Task-Incremental SL."""

    # In this example, `y_pred` is the predicted class for each image.
    y_pred: Tensor


@dataclass(frozen=True)
class Rewards(Setting.Rewards):
    """The rewards/labels returned by an Environment in Task-Incremental SL."""

    # In this example, `y` is the true label for each image.
    y: torch.LongTensor


# Type variables, just for fun. The Environment and Model classes are generics of these.
ObservationType = TypeVar("ObservationType", bound=Observations)
ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)


class Environment(Setting.Environment):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        observation_space: Space[ObservationType],
        action_space: Space[ActionType],
        reward_space: Space[RewardType],
        num_workers: int,
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

    def __iter__(self) -> Iterator[Tuple[ObservationType, RewardType]]:  # type: ignore
        for x, y, t in super().__iter__():
            observations = self.observation_space.dtype(x=x, task_labels=t)
            rewards = self.reward_space.dtype(y=y)
            # Save the 'rewards' (image labels) in the queue, as mentioned above.
            yield observations, rewards


class TestEnvironment(Environment, Setting.TestEnvironment):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        observation_space: Space[ObservationType],
        action_space: Space[ActionType],
        reward_space: Space[RewardType],
        num_workers: int,
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
        # NOTE: PyTorch-Lightning adds a kind of `with_is_last` fn around the dataloader iterator,
        # which consumes one batch of data. This (usually) causes a delay between the batch of
        # observations yielded by the iterator and the actions received by the `send` method.
        # To address this, here we store the rewards temporarily in a small doubly-ended queue, so
        # that we can re-align the actions and rewards inside the `send` method.
        self._reward_queue: Deque[RewardType] = deque(maxlen=2)
        self._metrics: List[float] = []

    def __iter__(self) -> Iterator[Tuple[ObservationType, Optional[RewardType]]]:
        self._reward_queue.clear()
        for observations, rewards in super().__iter__():
            if len(self._reward_queue) == self._reward_queue.maxlen:
                raise RuntimeError(
                    f"The reward queue should never be full. This means that more than "
                    f"{self._reward_queue.maxlen} observations were pulled from the "
                    f"environment without receiving an action in the `send` method. "
                    f"Make sure that you send an action to the test environment after each "
                    f"consumed batch, using the `send` method. "
                )
            self._reward_queue.append(rewards)
            yield observations, None

    def send(self, actions: ActionType) -> RewardType:
        rewards = self._reward_queue.popleft()
        self._metrics.append(self.get_metric(actions=actions, rewards=rewards))
        return rewards

    def get_metric(
        self,
        actions: TaskIncrementalSetting.Actions,
        rewards: TaskIncrementalSetting.Rewards,
    ) -> float:
        # NOTE: For now in this example we only consider SL, but would be different in RL!
        if (
            actions.y_pred.is_floating_point()
            or actions.y_pred.shape != rewards.y.shape
        ):
            # y_pred contains the logits
            y_pred = actions.y_pred.argmax(-1)
        else:
            # y_pred contains the predicted labels
            y_pred = actions.y_pred
        predicted_labels = y_pred.to(rewards.y.device)
        true_labels = rewards.y
        return float((predicted_labels == true_labels).sum()) / len(true_labels)


class TaskIncrementalSetting(Setting):
    """Example of a Setting class for Task-Incremental Supervised Learning.

    In this Setting, the Method is trained on different tasks, in sequence, and we
    measure the performance over all tasks learned so far.
    """

    # Class variable holding the available datasets for this setting.
    available_datasets: ClassVar[Dict[str, Type[Dataset]]] = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }

    # Save the types defined above as attributes on the Setting class itself.
    Environment: ClassVar[Type[Environment]] = Environment
    TestEnvironment: ClassVar[Type[TestEnvironment]] = TestEnvironment
    Observations: ClassVar[Type[Observations]] = Observations
    Actions: ClassVar[Type[Actions]] = Actions
    Rewards: ClassVar[Type[Rewards]] = Rewards

    @dataclass
    class Results(Setting.Results):
        """The results of applying a Method to the TaskIncrementalSetting."""

        # 2d matrix containing, for any index (i, j) the test "performance" on task `j`
        # after having learned tasks `0, ..., i`. For example, `transfer_matrix[1][1]`
        # is the performance on task 1 after having trained on task 0 and then task 1.
        # NOTE: In this example, the items could correspond to the classification
        # accuracy.
        transfer_matrix: Sequence[Sequence[float]]
        # List containing the performance on the training environment after learning
        # each task.
        training_performance: Sequence[float]

        @property
        def objective(self) -> float:
            """
            In this particular setting (Task-Incremental learning), we measure the final
            performance on all previous tasks.

            Other settings could have a different objective however! For example, in
            some settings, it might be important to measure the online training
            performance, to see how data-efficient the Method is. Other settings might
            care about forward transfer, etc.
            Such settings (possibly subclasses of this one) would then override this and
            change how the objective is computed as a function of the Results.
            """
            return float(np.mean(self.transfer_matrix[-1]))

    def __init__(
        self,
        dataset: str,
        nb_tasks: int,
        data_path: Path = Path("data"),
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
    ):
        """Create an instance of this Setting, using a particular dataset."""
        self.dataset = dataset
        # The type of dataset to use.
        self.dataset_class: Type[Dataset] = self.available_datasets[self.dataset]
        super().__init__()
        # Path where the data should be downloaded.
        self.data_path = data_path
        # Number of tasks to create in total.
        self.nb_tasks = nb_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        # List of datasets for each task.
        self.train_datasets: List[Dataset] = []
        self.val_datasets: List[Dataset] = []
        self.test_datasets: List[Dataset] = []

        self.current_task_id: int = 0

        self.num_classes: int
        self.dims: Tuple[int, ...]
        if "mnist" in self.dataset:
            self.dims = (1, 28, 28)
            self.num_classes = 10
        elif "cifar" in self.dataset:
            self.dims = (3, 32, 32)
            self.num_classes = int(self.dataset.split(sep="cifar", maxsplit=1)[1])
        else:
            raise NotImplementedError(
                "This example only considers the mnist and cifar variants."
            )

        self.observation_space: Space[
            TaskIncrementalSetting.Observations
        ] = TypedDictSpace(
            x=spaces.Box(0, 1, self.dims, dtype=np.float32),
            task_labels=spaces.Discrete(n=self.nb_tasks),
            dtype=self.Observations,
        )
        self.action_space: Space[TaskIncrementalSetting.Actions] = TypedDictSpace(
            y_pred=spaces.Discrete(self.num_classes), dtype=self.Actions
        )
        self.reward_space: Space[TaskIncrementalSetting.Rewards] = TypedDictSpace(
            y=spaces.Discrete(self.num_classes), dtype=self.Rewards
        )

        # The 'results' object that is created during `apply`. This could be used as the Setting's
        # 'state' the run so far. The Method would save its own state however it wants to.
        self._results: Optional[TaskIncrementalSetting.Results] = None

    def prepare_data(self) -> None:
        """Download the data required to create the datasets of each task."""
        _ = self.dataset_class(data_path=self.data_path, download=True, train=True)
        _ = self.dataset_class(data_path=self.data_path, download=True, train=False)

    def setup(self, stage: Optional[Stage] = None) -> None:
        """Create the datasets for each task (not included here for brevity.)"""
        # NOTE: Not included here, just to try and keep this short:
        if all([self.train_datasets, self.val_datasets, self.test_datasets]):
            # Datasets for each task were already created, no need to re-create them.
            return
        train_val_dataset = self.dataset_class(
            data_path=self.data_path, download=False, train=True
        )
        test_dataset = self.dataset_class(
            data_path=self.data_path, download=False, train=False
        )
        train_cl_scenario = ClassIncremental(
            train_val_dataset,
            nb_tasks=self.nb_tasks,
            transformations=self.train_transforms,
        )
        test_cl_scenario = ClassIncremental(
            test_dataset,
            nb_tasks=self.nb_tasks,
            transformations=self.test_transforms,
        )
        for train_taskset in train_cl_scenario:
            train_dataset, val_dataset = split_train_val(
                train_taskset, val_split=self.val_split
            )
            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)

        for test_taskset in test_cl_scenario:
            self.test_datasets.append(test_taskset)

    def train_dataloader(self) -> TaskIncrementalSetting.Environment:
        """Returns the training environment/dataloader for the current task."""
        if not self._has_prepared_data:
            self.prepare_data()
        if not self._has_setup_fit:
            self.setup("fit")
        dataset = self.train_datasets[self.current_task_id]
        return self.Environment(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
        )

    def val_dataloader(self) -> Environment:
        """Returns the validation environment/dataloader for the current task."""
        if not self._has_prepared_data:
            self.prepare_data()
        if not self._has_setup_validate:
            self.setup("validate")

        dataset = self.val_datasets[self.current_task_id]
        return self.Environment(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
        )

    def test_dataloader(self) -> TestEnvironment:
        """Returns the testing environment/dataloader for the current task."""
        # NOTE: In Sequoia: The test dataloader is a bit different than the train/val
        # dataloaders: It doesn't give the "rewards" (e.g. image labels `y`) until after it
        # receives the "action" (e.g. predicted class `y_pred`) in its `send` method.
        dataset = self.test_datasets[self.current_task_id]
        return self.TestEnvironment(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
        )

    def apply(self, method: Method) -> Results:
        """Apply a Method onto this Setting, producing Results."""
        # 1. Give the Method a chance to configure itself before training/testing
        # begins.
        method.configure(setting=self)

        # 2. Create the `Results` that will be returned:
        # NOTE: This `Results` object can be considered as the 'state' of the Setting.
        # It would be easy to save/restore a run that didn't complete properly by modifying the for
        # loops below a bit.
        self._results = self.Results(
            transfer_matrix=np.zeros([self.nb_tasks, self.nb_tasks]),
            training_performance=np.zeros(self.nb_tasks),
        )

        # 3. Start the "Main loop":
        # In this example, we train the Method on each task (by calling `fit`), and then
        # evaluate on all past and future tasks.

        for task_id in range(self.nb_tasks):
            self.current_task_id = task_id
            # 4. Train on task `i`

            # Let the Method know it is in the 'training' portion of the loop:
            method.training = True
            # In this example (Task-Incremental Learning), we always have access to the
            # task labels, therefore we inform the method of the task switch:
            method.on_task_switch(task_id=task_id)

            # Let the Method train and validate itself with the environments for the
            # current task.
            train_env = self.train_dataloader()
            valid_env = self.val_dataloader()

            # NOTE: If you wanted to, you could apply some `gym.Wrapper`s on top of the train or
            # valid envs, to, for example: transform the observations, limit the number of steps or
            # epochs, invoke a callback when a given step is reached, render the env,
            # accumulate/log metrics, etc etc.
            # for train_wrapper in self.train_wrappers:
            #     train_env = train_wrapper(train_env)

            method.fit(train_env=train_env, valid_env=valid_env)

            # NOTE: Optionally, if we were in a Setting where online training
            # performance was important, we could add a wrapper on the train environment
            # to, for example, measure the accuracy at each 'step' during the first
            # training epoch, and add this to the Results that are to be returned.
            # training_accuracy = train_env.get_performance()
            # self._results.training_performance.append(training_accuracy)

            # 5. Test loop: Evaluate the Method on all past and future tasks:
            method.training = False
            for test_task_id in range(self.nb_tasks):
                self.current_task_id = test_task_id

                # In this example (Task-Incremental Learning), we always have access to the
                # task labels, therefore we inform the method of the task switch:
                method.on_task_switch(task_id=test_task_id)

                # Run the 'test loop'.
                # In this example we allow the Method to iterate through the test
                # environment, and we ask the test environment for the performance it
                # observed afterwards.
                # NOTE: This is way better than asking the Method to evaluate itself! :P
                # NOTE: In this example, we assume that the Method has implemented a `test` method,
                # while in the actual API, if the Method doesn't have it, we perform the test loop
                # manually by calling `method.get_actions`.
                test_env = self.test_dataloader()
                method.test(test_env)
                task_test_acc = test_env.get_performance()

                # Put the average performance (accuracy) in the transfer matrix (at index
                # `(task_id, test_task_id)`).
                self._results.transfer_matrix[task_id][test_task_id] = task_test_acc

        return self._results


@dataclass
class TrainerConfig:
    """Configuration options for the `pl.Trainer`."""

    max_epochs: int = 3
    gpus: int = field(default_factory=torch.cuda.device_count)


class ExampleMethod(Method[TaskIncrementalSetting]):
    """Pseudo-code / Example of a Method that uses PyTorch-Lightning.

    Uses a `pl.LightningModule` as a model, and a `pl.Trainer` to train it on the Environments of
    Sequoia.
    """

    def __init__(self, trainer_config: TrainerConfig = None) -> None:
        """Create the Method."""
        # The model that will be trained.
        self.trainer_config = trainer_config or TrainerConfig()
        self.model: ExampleModel
        self.trainer: Trainer
        self._training: bool = True

    def configure(self, setting: TaskIncrementalSetting) -> None:
        """Called by the Setting to give the Method the opportunity to customize itself
        before training begins.
        """
        # Create the Model, using properties of the Setting such as the
        # observation_space, action_space, number of tasks, etc.
        self.model = ExampleModel(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
        )
        # if torch.cuda.is_available() and self.trainer_config.gpus > 0:
        #     self.model = self.model.to("cuda")

    def fit(
        self,
        train_env: TaskIncrementalSetting.Environment,
        valid_env: TaskIncrementalSetting.Environment,
    ) -> None:
        # NOTE: Currently creates a new Trainer for each Task, but you could also instead use the
        # same trainer for all tasks. That's totally up to the Method to figure out.
        self.trainer = Trainer(**asdict(self.trainer_config))
        self.trainer.fit(
            self.model, train_dataloader=train_env, val_dataloaders=valid_env
        )

    def test(self, test_env: DataLoader) -> None:
        """Iterate over the test environment ourselves.
        The test environment will report back our performance to the Setting.
        """
        # NOTE: Using `ckpt_path=None` so we use the current weights.
        self.trainer.test(self.model, test_env, ckpt_path=None)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Called by the Setting when we're encountering a task boundary.
        If the Setting doesn't provide task identity during the current phase, then
        `task_id` will be `None`.
        """
        # NOTE: You could do whatever you want here to help your model consolidate its
        # knowledge, otherwise it might start forgetting! :)


# NOTE: We're not going to re-implement all of the Model used by the Method above here. Instead, we
# just import the Model from the existing PL example. The fact that this works fine, even though
# we re-created the environments above is a good sign too! :)
from examples.basic.pl_example import Model


class ExampleModel(Model, Generic[ObservationType, ActionType, RewardType]):
    """Skeleton/pseudo-code for a Model, showing the main methods as well as their types.

    The Model's job is simple: It has to produce Actions (predictions) for a given set of
    Observations (samples).

    Additionally, since we use PyTorch-Lightning in this example, it has to implement the
    training_step.

    This is just a skeleton/pseudocode to give you an idea of the API. If you want to see what an
    actual Model would look like in Sequoia, take a look at the pl_example.py file at this url:
    https://github.com/lebrice/Sequoia/blob/master/examples/basic/pl_example.py
    """

    def __init__(
        self,
        observation_space: Space[ObservationType],
        action_space: Space[ActionType],
        reward_space: Space[RewardType],
    ):
        super().__init__(input_space=observation_space, output_space=action_space)
        self.reward_space = reward_space

    def forward(self, obs: ObservationType) -> ActionType:  # type: ignore
        return super().forward(obs)

    def training_step(self, batch: Tuple[ObservationType, RewardType], *args, **kwargs):  # type: ignore
        return super().training_step(batch, *args, **kwargs)

    def validation_step(self, batch: Tuple[ObservationType, RewardType], *args, **kwargs):  # type: ignore
        return super().validation_step(batch, *args, **kwargs)

    def test_step(self, batch: Tuple[ObservationType, Optional[RewardType]], *args, **kwargs):  # type: ignore
        return super().test_step(batch, *args, **kwargs)

    # NOTE: Omitting the `[training/val/test]_step_end` in this example, but they would be added
    # if we want to support multi-GPU training in this Model.


def main() -> None:
    # NOTE: This isn't a functional example, it's just an illustration, but this is how
    # you would apply a Method to a Setting to get Results.
    setting = TaskIncrementalSetting(dataset="mnist", nb_tasks=5)
    method = ExampleMethod()
    results: TaskIncrementalSetting.Results = setting.apply(method)
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
