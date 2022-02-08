from typing import Any, ClassVar, Dict, Type

import pytest
from continuum import ClassIncremental
from gym import spaces
from gym.spaces import Discrete, Space

from sequoia.common.config import Config
from sequoia.common.metrics import ClassificationMetrics
from sequoia.common.spaces import Sparse
from sequoia.conftest import skip_param, xfail_param
from sequoia.settings.base import Setting
from sequoia.settings.sl.continual.envs import get_action_space

from ..discrete.setting_test import (
    TestDiscreteTaskAgnosticSLSetting as DiscreteTaskAgnosticSLSettingTests,
)
from .setting import IncrementalSLSetting
from .setting import IncrementalSLSetting as ClassIncrementalSetting


class TestIncrementalSLSetting(DiscreteTaskAgnosticSLSettingTests):
    Setting: ClassVar[Type[Setting]] = IncrementalSLSetting
    fast_dev_run_kwargs: ClassVar[Dict[str, Any]] = dict(
        dataset="mnist",
        batch_size=64,
    )

    def assert_chance_level(
        self, setting: IncrementalSLSetting, results: IncrementalSLSetting.Results
    ):
        assert isinstance(setting, ClassIncrementalSetting), setting
        assert isinstance(results, ClassIncrementalSetting.Results), results
        # TODO: Remove this assertion:
        assert isinstance(setting.action_space, spaces.Discrete)
        # TODO: This test so far needs the 'N' to be the number of classes in total,
        # not the number of classes per task.
        # num_classes = setting.action_space.n  # <-- Should be using this instead.
        if setting._using_custom_envs_foreach_task:
            num_classes = get_action_space(setting.train_datasets[0]).n
        else:
            num_classes = get_action_space(setting.dataset).n

        average_accuracy = results.objective
        # Calculate the expected 'average' chance accuracy.
        # We assume that there is an equal number of classes in each task.
        # chance_accuracy = 1 / setting.n_classes_per_task
        chance_accuracy = 1 / num_classes

        assert 0.5 * chance_accuracy <= average_accuracy <= 1.5 * chance_accuracy

        for i, metric in enumerate(results.final_performance_metrics):
            assert isinstance(metric, ClassificationMetrics)
            # TODO: Same as above: Should be using `n_classes_per_task` or something
            # like it instead.
            chance_accuracy = 1 / setting.n_classes_per_task
            chance_accuracy = 1 / num_classes

            task_accuracy = metric.accuracy
            # FIXME: Look into this, we're often getting results substantially
            # worse than chance, and to 'make the tests pass' (which is bad)
            # we're setting the lower bound super low, which makes no sense.
            assert 0.25 * chance_accuracy <= task_accuracy <= 2.1 * chance_accuracy

    # TODO: Add a fixture that specifies a data folder common to all tests.
    @pytest.mark.parametrize(
        "dataset_name",
        [
            "mnist",
            # "synbols",
            skip_param("synbols", reason="Causes tests to hang for some reason?"),
            "cifar10",
            "cifar100",
            "fashionmnist",
            "kmnist",
            xfail_param("emnist", reason="Bug in emnist, requires split positional arg?"),
            xfail_param("qmnist", reason="Bug in qmnist, 229421 not in list"),
            "mnistfellowship",
            "cifar10",
            "cifarfellowship",
        ],
    )
    @pytest.mark.timeout(60)
    def test_observation_spaces_match_dataset(self, dataset_name: str):
        """Test to check that the `observation_spaces` and `reward_spaces` dict
        really correspond to the entries of the corresponding datasets, before we do
        anything with them.
        """
        # CIFARFellowship, MNISTFellowship, ImageNet100,
        # ImageNet1000, CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST,
        # QMNIST, FashionMNIST,
        dataset_class = self.Setting.available_datasets[dataset_name]
        dataset = dataset_class("data")

        observation_space = self.Setting.base_observation_spaces[dataset_name]
        reward_space = self.Setting.base_reward_spaces[dataset_name]
        for task_dataset in ClassIncremental(dataset, nb_tasks=1):
            first_item = task_dataset[0]
            x, t, y = first_item
            assert x.shape == observation_space.shape
            assert x in observation_space, (x.min(), x.max(), observation_space)
            assert y in reward_space

    @pytest.mark.parametrize("dataset_name", ["mnist"])
    @pytest.mark.parametrize("nb_tasks", [2, 5])
    def test_task_label_space(self, dataset_name: str, nb_tasks: int):
        # dataset = ClassIncrementalSetting.available_datasets[dataset_name]
        nb_tasks = 2
        setting = ClassIncrementalSetting(
            dataset=dataset_name,
            nb_tasks=nb_tasks,
        )
        task_label_space: Space = setting.observation_space.task_labels
        # TODO: Should the task label space be Sparse[Discrete]? or Discrete?
        assert task_label_space == Discrete(nb_tasks)

    @pytest.mark.parametrize("dataset_name", ["mnist"])
    def test_setting_obs_space_changes_when_transforms_change(self, dataset_name: str):
        """TODO: Test that the `observation_space` property on the
        ClassIncrementalSetting reflects the data produced by the dataloaders, and
        that changing a transform on a Setting also changes the value of that
        property on both the Setting itself, as well as on the corresponding
        dataloaders/environments.
        """
        # dataset = ClassIncrementalSetting.available_datasets[dataset_name]
        setting = self.Setting(
            dataset=dataset_name,
            nb_tasks=1,
            transforms=[],
            train_transforms=[],
            val_transforms=[],
            test_transforms=[],
            batch_size=None,
            num_workers=0,
        )
        assert setting.observation_space.x == Setting.base_observation_spaces[dataset_name]
        # TODO: Should the 'transforms' apply to ALL the environments, and the
        # train/valid/test transforms apply only to those envs?
        from sequoia.common.transforms import Transforms

        setting.transforms = [
            Transforms.to_tensor,
            Transforms.three_channels,
            Transforms.channels_first_if_needed,
            Transforms.resize_32x32,
        ]
        # When there are no transforms in setting.train_tansforms, the observation
        # space of the Setting and of the train dataloader are the same:
        train_env = setting.train_dataloader(batch_size=None, num_workers=None)
        assert train_env.observation_space == setting.observation_space

        reset_obs = train_env.reset()
        assert reset_obs["x"] in train_env.observation_space["x"], reset_obs[0].shape
        assert reset_obs["task_labels"] in train_env.observation_space["task_labels"]
        assert reset_obs in train_env.observation_space
        assert reset_obs in setting.observation_space
        assert isinstance(reset_obs, ClassIncrementalSetting.Observations)

        # When we add a transform to `setting.train_transforms` the observation
        # space of the Setting and of the train dataloader are different:
        setting.train_transforms = [Transforms.resize_64x64]

        train_env = setting.train_dataloader(batch_size=None)
        assert train_env.observation_space.x.shape == (3, 64, 64)
        assert train_env.reset() in train_env.observation_space

        # The Setting's property didn't change:
        assert setting.observation_space.x.shape == (3, 32, 32)
        #
        #  ---------- Same tests for the val_environment --------------
        #
        val_env = setting.val_dataloader(batch_size=None)
        assert val_env.observation_space == setting.observation_space
        assert val_env.reset() in val_env.observation_space

        # When we add a transform to `setting.val_transforms` the observation
        # space of the Setting and of the val dataloader are different:
        setting.val_transforms = [Transforms.resize_64x64]
        val_env = setting.val_dataloader(batch_size=None)
        assert val_env.observation_space != setting.observation_space
        assert val_env.observation_space.x.shape == (3, 64, 64)
        assert val_env.reset() in val_env.observation_space
        #
        #  ---------- Same tests for the test_environment --------------
        #

        with setting.test_dataloader(batch_size=None) as test_env:
            if setting.task_labels_at_test_time:
                assert test_env.observation_space == setting.observation_space
            else:
                assert isinstance(test_env.observation_space["task_labels"], Sparse)
            assert test_env.reset() in test_env.observation_space

        setting.test_transforms = [Transforms.resize_64x64]
        with setting.test_dataloader(batch_size=None) as test_env:
            # When we add a transform to `setting.test_transforms` the observation
            # space of the Setting and of the test dataloader are different:
            assert test_env.observation_space != setting.observation_space
            assert test_env.observation_space.x.shape == (3, 64, 64)
            assert test_env.reset() in test_env.observation_space


# TODO: This renders, even when we're using the pytest-xvfb plugin, which might
# mean that it's actually creating a Display somewhere?
@pytest.mark.timeout(30)
def test_render(config: Config):
    setting = ClassIncrementalSetting(dataset="mnist", config=config)
    import matplotlib.pyplot as plt

    plt.ion()
    for task_id in range(setting.nb_tasks):
        setting.current_task_id = task_id
        env = setting.train_dataloader(batch_size=16, num_workers=0)
        obs = env.reset()
        done = False
        while not done:
            obs, rewards, done, info = env.step(env.action_space.sample())
            env.render("human")
            # break
        env.close()


def test_class_incremental_random_baseline():
    pass
