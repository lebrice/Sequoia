from collections import Counter
from typing import Any, ClassVar, Dict, Type
import functools
import gym
import pytest
from sequoia.common.config import Config
from sequoia.methods import RandomBaselineMethod
from sequoia.settings import Setting
from sequoia.settings.base.setting_test import SettingTests
from pathlib import Path

from .setting import ContinualSLSetting, smooth_task_boundaries_concat


from continuum.tasks import TaskSet, concat
from continuum.datasets import MNIST
from continuum.scenarios import ClassIncremental
from sequoia.common.config import Config
from sequoia.settings.sl.continual.setting import shuffle


def test_shuffle(config: Config):
    dataset = MNIST(data_path=config.data_dir, train=True)
    cl_dataset = concat(ClassIncremental(dataset, increment=2))
    shuffled_dataset = shuffle(cl_dataset)
    assert (shuffled_dataset._y != cl_dataset._y).sum() > len(cl_dataset) / 2
    assert (shuffled_dataset._t != cl_dataset._t).sum() > len(cl_dataset) / 2
    # assert False, list(zip(shuffled_dataset._t, cl_dataset._t, shuffled_dataset._y, cl_dataset._y))[:10]


class TestContinualSLSetting(SettingTests):
    Setting: ClassVar[Type[Setting]] = ContinualSLSetting

    # The kwargs to be passed to the Setting when we want to create a 'short' setting.
    fast_dev_run_kwargs: ClassVar[Dict[str, Any]] = dict(
        dataset="mnist", batch_size=64,
    )

    # @pytest.fixture
    # def setting_kwargs(self):

    def test_shared_action_space(self, config: Config):
        kwargs = dict(dataset="mnist", config=config)
        if (
            isinstance(self.Setting, functools.partial)
            and not self.Setting.args[0].shared_action_space
        ):
            # NOTE: This `self.Setting` being a partial instead of a Setting class only
            # happens in the tests for the SettingProxy. 
            kwargs.update(shared_action_space=True)
        elif not self.Setting.shared_action_space:
            kwargs.update(shared_action_space=True)

        setting = self.Setting(**kwargs)
        y_counter = Counter()
        t_counter = Counter()
        test_env = setting.test_dataloader(batch_size=128, num_workers=4)
        for obs, rewards in test_env:
            if rewards is None:
                rewards = test_env.send(test_env.action_space.sample())

            y = rewards.y.tolist()
            t = (
                obs.task_labels.tolist()
                if obs.task_labels is not None
                else [None for _ in range(obs.x.shape[0])]
            )
            y_counter.update(y)
            t_counter.update(t)

        # This is what you get with mnist, with the default class ordering:
        # if setting.known_task_boundaries_at_train_time:
        #     # Only the first task of mnist, in this case.
        #     assert y_counter == {1: 6065, 0: 5534}

        assert y_counter == {0: 4926, 1: 5074}
        if setting.task_labels_at_test_time:
            assert t_counter == {0: 2115, 1: 2042, 3: 1986, 4: 1983, 2: 1874}
        else:
            assert t_counter == {None: 10_000}
        # assert t_counter

        # Full Train envs:
        # assert y_counter == {1: 27456, 0: 26546}
        # assert False, c

    def test_only_one_epoch(self, config: Config):
        setting = self.Setting(dataset="mnist", config=config)
        train_env = setting.train_dataloader(batch_size=100, num_workers=4)

        for _ in train_env:
            pass
        if not setting.known_task_boundaries_at_train_time:
            assert train_env.is_closed()
            with pytest.raises(gym.error.ClosedEnvironmentError):
                for _ in train_env:
                    pass

    @pytest.mark.no_xvfb
    @pytest.mark.timeout(20)
    @pytest.mark.skipif(
        not Path("temp").exists(),
        reason="Need temp dir for saving the figure this test creates.",
    )
    def test_show_distributions(self, config: Config):
        setting = self.Setting(dataset="mnist", config=config)

        import matplotlib.pyplot as plt
        from functools import partial

        fig, axes = plt.subplots(2, 3)
        name_to_env_fn = {
            "train": setting.train_dataloader,
            "valid": setting.val_dataloader,
            "test": setting.test_dataloader,
        }
        for i, (name, env_fn) in enumerate(name_to_env_fn.items()):
            env = env_fn(batch_size=100, num_workers=4)

            y_counters: List[Counter] = []
            t_counters: List[Counter] = []

            for obs, reward in env:
                if reward is None:
                    reward = env.send(env.action_space.sample())
                y = reward.y.cpu().numpy()
                t = obs.task_labels
                if t is None:
                    t = [None for _ in y]
                y_count = Counter(y.tolist())
                t_count = Counter(t)

                y_counters.append(y_count)
                t_counters.append(t_count)

            classes = list(set().union(*y_counters))
            task_ids = list(set().union(*t_counters))

            nb_classes = len(classes)
            x = np.arange(len(env))

            for label in range(nb_classes):
                y = [y_counter.get(label) for y_counter in y_counters]
                axes[0, i].plot(x, y, label=f"y={label}")
            axes[0, i].legend()
            axes[0, i].set_title(f"{name} y")
            axes[0, i].set_xlabel("Batch index")
            axes[0, i].set_ylabel("Count in batch")

            for task_id in task_ids:
                y = [t_counter.get(task_id) for t_counter in t_counters]
                axes[1, i].plot(x, y, label=f"task_id={task_id}")
            axes[1, i].legend()
            axes[1, i].set_title(f"{name} task_id")
            axes[1, i].set_xlabel("Batch index")
            axes[1, i].set_ylabel("Count in batch")

        plt.legend()

        Path("temp").mkdir(exist_ok=True)
        fig.set_size_inches((6, 4), forward=False)
        plt.savefig(f"temp/{self.Setting.__name__}.png")
        # plt.waitforbuttonpress(10)
        # plt.show()


from typing import List, Tuple

import numpy as np
import pytest
from continuum import TaskSet
from torch.utils.data import DataLoader


@pytest.mark.timeout(30)
@pytest.mark.no_xvfb
def test_concat_smooth_boundaries(config: Config):
    from continuum.datasets import MNIST
    from continuum.scenarios import ClassIncremental
    from continuum.tasks import split_train_val

    dataset = MNIST(config.data_dir, download=True, train=True)
    scenario = ClassIncremental(dataset, increment=2,)

    print(f"Number of classes: {scenario.nb_classes}.")
    print(f"Number of tasks: {scenario.nb_tasks}.")

    train_datasets = []
    valid_datasets = []
    for task_id, train_taskset in enumerate(scenario):
        train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
        train_datasets.append(train_taskset)
        valid_datasets.append(val_taskset)

    # train_datasets = [Subset(task_dataset, np.arange(20)) for task_dataset in train_datasets]
    train_dataset = smooth_task_boundaries_concat(train_datasets, seed=123)

    xs = np.arange(len(train_dataset))
    y_counters: List[Counter] = []
    t_counters: List[Counter] = []
    dataloader = DataLoader(train_dataset, batch_size=100, shuffle=False)

    for x, y, t in dataloader:
        y_count = Counter(y.tolist())
        t_count = Counter(t.tolist())

        y_counters.append(y_count)
        t_counters.append(t_count)

    classes = list(set().union(*y_counters))
    nb_classes = len(classes)
    x = np.arange(len(dataloader))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2)
    for label in range(nb_classes):
        y = [y_counter.get(label) for y_counter in y_counters]
        axes[0].plot(x, y, label=f"class {label}")
    axes[0].legend()
    axes[0].set_title("y")
    axes[0].set_xlabel("Batch index")
    axes[0].set_ylabel("Count in batch")

    for task_id in range(scenario.nb_tasks):
        y = [t_counter.get(task_id) for t_counter in t_counters]
        axes[1].plot(x, y, label=f"Task id {task_id}")
    axes[1].legend()
    axes[1].set_title("task_id")
    axes[1].set_xlabel("Batch index")
    axes[1].set_ylabel("Count in batch")

    plt.legend()
    # plt.waitforbuttonpress(10)
    # plt.show()
