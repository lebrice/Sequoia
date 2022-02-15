import itertools
import math
from typing import *

import pytest

from sequoia.common.config import Config
from sequoia.settings.assumptions.incremental_test import OtherDummyMethod
from sequoia.utils.logging_utils import get_logger

from ..incremental.setting_test import TestIncrementalSLSetting as IncrementalSLSettingTests
from .setting import TaskIncrementalSLSetting

logger = get_logger(__name__)


class TestTaskIncrementalSLSetting(IncrementalSLSettingTests):
    Setting: ClassVar[Type[Setting]] = TaskIncrementalSLSetting
    fast_dev_run_kwargs: ClassVar[Dict[str, Any]] = dict(
        dataset="mnist",
        batch_size=64,
    )


def check_only_right_classes_present(setting: TaskIncrementalSLSetting):
    """Checks that only the classes within each task are present.

    TODO: This should be refactored to be based more on the reward space.
    """
    assert setting.task_labels_at_test_time and setting.task_labels_at_test_time

    for i in range(setting.nb_tasks):
        setting.current_task_id = i
        batch_size = 5
        train_loader = setting.train_dataloader(batch_size=batch_size)

        # get the classes in the current task:
        task_classes = setting.task_classes(i, train=True)

        for j, (observations, rewards) in enumerate(itertools.islice(train_loader, 100)):
            x = observations.x
            t = observations.task_labels

            if setting.task_labels_at_train_time:
                assert t is not None

            y = rewards.y
            print(i, j, y, t)
            y_in_task_classes = [y_i in task_classes for y_i in y.tolist()]
            assert all(y_in_task_classes)
            assert x.shape == (batch_size, 3, 28, 28)
            x = x.permute(0, 2, 3, 1)[0]
            assert x.shape == (28, 28, 3)

            reward = train_loader.send([4 for _ in range(batch_size)])
            if rewards is not None:
                # IF we send somethign to the env, then it should give back the same
                # labels as for the last batch.
                assert (reward.y == rewards.y).all()

        train_loader.close()

        valid_loader = setting.val_dataloader(batch_size=batch_size)
        for j, (observations, rewards) in enumerate(itertools.islice(valid_loader, 100)):
            x = observations.x
            t = observations.task_labels

            if setting.monitor_training_performance:
                assert rewards is None

            if setting.task_labels_at_train_time:
                assert t is not None

            y = rewards.y
            print(i, j, y, t)
            y_in_task_classes = [y_i in task_classes for y_i in y.tolist()]
            assert all(y_in_task_classes)
            assert x.shape == (batch_size, 3, 28, 28)
            x = x.permute(0, 2, 3, 1)[0]
            assert x.shape == (28, 28, 3)

            reward = valid_loader.send(valid_loader.action_space.sample())
            if rewards is not None:
                # IF we send somethign to the env, then it should give back the same
                # labels as for the last batch.
                assert (reward.y == rewards.y).all()

        valid_loader.close()

        # FIXME: get the classes in the current task, at test-time.
        task_classes = list(range(setting.reward_space.n))

        test_loader = setting.test_dataloader(batch_size=batch_size)
        assert not test_loader.unwrapped._hide_task_labels
        for j, (observations, rewards) in enumerate(itertools.islice(test_loader, 100)):
            x = observations.x
            t = observations.task_labels
            if setting.task_labels_at_test_time:
                assert t is not None

            if rewards is None:
                rewards = test_loader.send(test_loader.action_space.sample())
                assert rewards is not None
                assert rewards.y is not None

            y = rewards.y
            print(i, j, y, t)
            y_in_task_classes = [y_i in task_classes for y_i in y.tolist()]
            assert all(y_in_task_classes)
            assert x.shape == (batch_size, 3, 28, 28)
            x = x.permute(0, 2, 3, 1)[0]
            assert x.shape == (28, 28, 3)

        test_loader.close()


def test_task_incremental_mnist_setup():
    setting = TaskIncrementalSLSetting(
        dataset="mnist",
        increment=2,
        # BUG: When num_workers > 0, some of the tests hang, but only when running *all* the tests!
        # num_workers=0,
    )
    assert setting.task_labels_at_test_time and setting.task_labels_at_train_time
    setting.prepare_data(data_dir="data")
    setting.setup()
    check_only_right_classes_present(setting)


@pytest.mark.xfail(
    reason=(
        "TODO: Continuum actually re-labels the images to 0-10, regardless of the "
        "class order. The actual images are ok though."
    )
)
def test_task_incremental_mnist_setup_reversed_class_order():
    setting = TaskIncrementalSLSetting(
        dataset="mnist",
        nb_tasks=5,
        class_order=list(reversed(range(10))),
        # num_workers=0,
    )
    assert setting.task_labels_at_train_time and setting.task_labels_at_test_time
    assert (
        setting.known_task_boundaries_at_train_time and setting.known_task_boundaries_at_test_time
    )
    setting.prepare_data(data_dir="data")
    setting.setup()
    check_only_right_classes_present(setting)


def test_class_incremental_mnist_setup_with_nb_tasks():
    setting = TaskIncrementalSLSetting(
        dataset="mnist",
        nb_tasks=2,
        num_workers=0,
    )
    assert setting.increment == 5
    setting.prepare_data(data_dir="data")
    setting.setup()
    assert len(setting.train_datasets) == 2
    assert len(setting.val_datasets) == 2
    assert len(setting.test_datasets) == 2
    check_only_right_classes_present(setting)


def test_action_space_always_matches_obs_batch_size(config: Config):
    """Make sure that the batch size in the observations always matches the action
    space provided to the `get_actions` method.

    ALSO:
    - Make sure that we get asked for actions for all the observations in the test set,
      even when there is a shorter last batch.
    - The total number of observations match the dataset size.
    """
    nb_tasks = 5
    # TODO: The `drop_last` argument seems to not be used correctly by the dataloaders / test loop.
    batch_size = 128

    # HUH why are we doing this here?
    setting = TaskIncrementalSLSetting(
        dataset="mnist",
        nb_tasks=nb_tasks,
        batch_size=batch_size,
        num_workers=4,
        monitor_training_performance=True,
        drop_last=False,
    )

    # 10_000 examples in the test dataset of mnist.
    total_samples = len(setting.test_dataloader().dataset)

    method = OtherDummyMethod()
    _ = setting.apply(method, config=config)

    # Multiply by nb_tasks because the test loop is ran after each training task.
    assert sum(method.batch_sizes) == total_samples * nb_tasks
    assert len(method.batch_sizes) == math.ceil(total_samples / batch_size) * nb_tasks
    if total_samples % batch_size == 0:
        assert set(method.batch_sizes) == {batch_size}
    else:
        assert set(method.batch_sizes) == {batch_size, total_samples % batch_size}
