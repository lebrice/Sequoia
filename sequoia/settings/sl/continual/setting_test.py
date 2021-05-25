from collections import Counter
from typing import Any, ClassVar, Dict, Type

import gym
import pytest
from sequoia.common.config import Config
from sequoia.methods import RandomBaselineMethod
from sequoia.settings import Setting
from sequoia.settings.base.setting_test import SettingTests

from .setting import ContinualSLSetting, smooth_task_boundaries_concat


class TestContinualSLSetting(SettingTests):
    Setting: ClassVar[Type[Setting]] = ContinualSLSetting

    # The kwargs to be passed to the Setting when we want to create a 'short' setting.
    fast_dev_run_kwargs: ClassVar[Dict[str, Any]] = dict(
        dataset="mnist", batch_size=64,
    )

    def test_shared_action_space(self, config: Config):
        setting = ContinualSLSetting(
            dataset="mnist", shared_action_space=True, config=config
        )
        c = Counter()
        train_env = setting.train_dataloader(batch_size=128, num_workers=4)
        for _, rewards in train_env:
            if rewards is None:
                rewards = train_env.send(train_env.action_space.sample())

            y = rewards.y.tolist()
            c.update(y)

        # This is what you get with mnist, with the default class ordering:
        assert c == {1: 27456, 0: 26546}
        # assert False, c

    def test_only_one_epoch(self, config: Config):
        setting = self.Setting(dataset="mnist", config=config)
        train_env = setting.train_dataloader(batch_size=100, num_workers=4)

        for _ in train_env:
            pass
        assert train_env.is_closed()
        with pytest.raises(gym.error.ClosedEnvironmentError):
            for _ in train_env:
                pass

    @pytest.mark.timeout(20)
    def test_show_distributions(self, config: Config):
        setting = self.Setting(dataset="mnist", config=config)
        y_counters: List[Counter] = []
        t_counters: List[Counter] = []
        train_env = setting.train_dataloader(batch_size=100, num_workers=4)
        for obs, reward in train_env:
            if reward is None:
                reward = train_env.send(train_env.action_space.sample())
            y = reward.y.cpu().numpy()
            t = obs.task_labels
            if t is None:
                t = [None for _ in y]
            y_count = Counter(y.tolist())
            t_count = Counter(t)

            y_counters.append(y_count)
            t_counters.append(t_count)

        classes = list(set().union(*y_counters))
        nb_classes = len(classes)
        x = np.arange(len(train_env))

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2)
        for label in range(nb_classes):
            y = [y_counter.get(label) for y_counter in y_counters]
            axes[0].plot(x, y, label=f"class {label}")
        axes[0].legend()
        axes[0].set_title("y")
        axes[0].set_xlabel("Batch index")
        axes[0].set_ylabel("Count in batch")

        for task_id in range(setting.nb_tasks):
            y = [t_counter.get(task_id) for t_counter in t_counters]
            axes[1].plot(x, y, label=f"Task id {task_id}")
        axes[1].legend()
        axes[1].set_title("task_id")
        axes[1].set_xlabel("Batch index")
        axes[1].set_ylabel("Count in batch")

        plt.legend()
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
