import functools
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar, Dict, Tuple, Type

import gym
import pytest
import torch
from continuum.datasets import MNIST
from continuum.scenarios import ClassIncremental
from continuum.tasks import TaskSet, concat
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split

from sequoia.common.config import Config
from sequoia.methods import RandomBaselineMethod
from sequoia.settings import Setting
from sequoia.settings.base.setting_test import SettingTests
from sequoia.settings.sl.continual.setting import shuffle

from .setting import ContinualSLSetting, random_subset, smooth_task_boundaries_concat
from .wrappers import ShowLabelDistributionWrapper


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
    # TODO: Transform this into a fixture instead.
    fast_dev_run_kwargs: ClassVar[Dict[str, Any]] = dict(
        dataset="mnist",
        batch_size=64,
    )

    @pytest.fixture(scope="session")
    def short_setting(self, session_config):
        kwargs = self.fast_dev_run_kwargs.copy()
        kwargs["config"] = session_config

        setting = self.Setting(**kwargs)
        setting.config = session_config
        setting.prepare_data()
        setting.setup()

        # Testing this out: Shortening the train datasets:
        setting.train_datasets = [
            random_subset(task_dataset, 100) for task_dataset in setting.train_datasets
        ]
        setting.val_datasets = [
            random_subset(task_dataset, 100) for task_dataset in setting.val_datasets
        ]
        setting.test_datasets = [
            random_subset(task_dataset, 100) for task_dataset in setting.test_datasets
        ]
        assert len(setting.train_datasets) == 5
        assert len(setting.val_datasets) == 5
        assert len(setting.test_datasets) == 5
        assert all(len(dataset) == 100 for dataset in setting.train_datasets)
        assert all(len(dataset) == 100 for dataset in setting.val_datasets)
        assert all(len(dataset) == 100 for dataset in setting.test_datasets)

        # Assert that calling setup doesn't overwrite the datasets.
        setting.setup()
        assert len(setting.train_datasets) == 5
        assert len(setting.val_datasets) == 5
        assert len(setting.test_datasets) == 5
        assert all(len(dataset) == 100 for dataset in setting.train_datasets)
        assert all(len(dataset) == 100 for dataset in setting.val_datasets)
        assert all(len(dataset) == 100 for dataset in setting.test_datasets)
        return setting

    def test_shared_action_space(self, config: Config):
        kwargs = self.fast_dev_run_kwargs.copy()
        kwargs["config"] = config
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
        test_env = setting.test_dataloader()
        for obs, rewards in test_env:
            if rewards is None:
                action = test_env.action_space.sample()
                # NOTE: On the last batch, the rewards might have a smaller batch size
                # than the action space.
                # TODO: Add tests to check that the envs can explicitly handle this, so
                # that we don't give the burden to the Method.
                rewards = test_env.send(action)

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

    def test_only_one_epoch(self, short_setting):
        setting = short_setting
        train_env = setting.train_dataloader()

        for _ in train_env:
            pass
        if not setting.known_task_boundaries_at_train_time:
            assert train_env.is_closed()
            with pytest.raises(gym.error.ClosedEnvironmentError):
                for _ in train_env:
                    pass
        else:
            assert not train_env.is_closed()

    @pytest.mark.no_xvfb
    @pytest.mark.timeout(20)
    @pytest.mark.skipif(
        not Path("temp").exists(),
        reason="Need temp dir for saving the figure this test creates.",
    )
    def test_show_distributions(self, config: Config):
        setting = self.Setting(dataset="mnist", config=config)
        figures_dir = Path("temp")

        from functools import partial

        import matplotlib.pyplot as plt

        # fig, axes = plt.subplots(2, 3)
        name_to_env_fn = {
            "train": setting.train_dataloader,
            "valid": setting.val_dataloader,
            "test": setting.test_dataloader,
        }
        # TODO: Maybe add these plots as part of the results for ContinualSL? How much
        # memory would actually be needed to store these here?
        for i, (name, env_fn) in enumerate(name_to_env_fn.items()):
            env = env_fn(batch_size=100, num_workers=4)
            env = ShowLabelDistributionWrapper(env, env_name=name)
            # Iterate through the env.
            for obs, rewards in env:
                if rewards is None:
                    rewards = env.send(env.action_space.sample())

            fig = env.make_figure()
            fig.set_size_inches((6, 4), forward=False)
            save_path = Path(f"{figures_dir}/{setting.get_name()}_{name}.png")
            save_path.parent.mkdir(exist_ok=True)
            fig.savefig(save_path)

        # plt.waitforbuttonpress(10)
        # plt.show()

    def test_passing_datasets_to_setting(self, config: Config):
        image_shape = (16, 16, 3)
        n_classes = 10
        datasets = [
            create_image_classification_dataset(image_shape=image_shape, n_classes=2, y_offset= i * 2)
            for i in range(5)
        ]
        train_datasets = []
        val_datasets = []
        test_datasets = []
        for dataset in datasets:
            n = len(dataset)
            n_train_val = int(n * 0.8)
            n_test = n - n_train_val
            n_train = int(n_train_val * 0.8)
            n_valid = n_train_val - n_train
            train_val_dataset, test_dataset = random_split(
                dataset, [n_train_val, n_test]
            )
            train_dataset, val_dataset = random_split(
                train_val_dataset, [n_train, n_valid]
            )

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)

        setting = self.Setting(
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            test_datasets=test_datasets,
            transforms=[],
            # train_transforms=[],
            # val_transforms=[],
            # test_transforms=[]
        )
        assert setting.train_datasets is train_datasets
        assert setting.val_datasets is val_datasets
        assert setting.test_datasets is test_datasets
        assert setting.nb_tasks == len(setting.train_datasets)
        assert setting.observation_space.x.shape == image_shape
        assert setting.reward_space.n == n_classes


def create_image_classification_dataset(
    image_shape: Tuple[int, ...],
    n_classes: int,
    n_samples_per_class: int = 100,
    y_offset: int = 0,
):
    """Copied and Adapted from
    https://github.com/ContinualAI/avalanche/blob/master/tests/unit_tests_utils.py
    """
    # n_classes = 10
    # image_shape = (16, 16, 3)
    # n_samples_per_class = 100
    n_features = np.prod(image_shape)
    dataset = make_classification(
        n_samples=n_classes * n_samples_per_class,
        n_classes=n_classes,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
    )
    x = torch.from_numpy(dataset[0]).reshape([-1, *image_shape]).float()
    y = torch.from_numpy(dataset[1]).long()
    # y_offset can be used to get [2,3] rather than [0,1] for instance.
    if y_offset:
        y += y_offset
    return TensorDataset(x, y)

    # train_X, test_X, train_y, test_y = train_test_split(
    #     X, y, train_size=0.6, shuffle=True, stratify=y)

    # train_dataset = TensorDataset(train_X, train_y)
    # test_dataset = TensorDataset(test_X, test_y)
    # return my_nc_benchmark


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
    scenario = ClassIncremental(
        dataset,
        increment=2,
    )

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
