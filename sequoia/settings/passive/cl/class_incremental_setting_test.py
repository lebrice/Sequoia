from pathlib import Path

import pytest
from continuum import InstanceIncremental
from sequoia.methods import RandomBaselineMethod
from sequoia.common.spaces import Sparse
from sequoia.common.gym_wrappers.convert_tensors import has_tensor_support

from .class_incremental_setting import (ClassIncrementalSetting,
                                        base_observation_spaces,
                                        dims_for_dataset, reward_spaces)

# TODO: Add a fixture that specifies a data folder common to all tests.


@pytest.mark.parametrize("dataset_name", ["mnist"])
def test_observation_spaces_match_dataset(dataset_name: str):
    """ Test to check that the `observation_spaces` and `reward_spaces` dict
    really correspond to the entries of the corresponding datasets, before we do
    anything with them.
    """
    # CIFARFellowship, MNISTFellowship, ImageNet100,
    # ImageNet1000, CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST,
    # QMNIST, FashionMNIST,
    dataset_class = ClassIncrementalSetting.available_datasets[dataset_name]
    dataset = dataset_class("data")

    observation_space = base_observation_spaces[dataset_name]
    reward_space = reward_spaces[dataset_name]
    for task_dataset in InstanceIncremental(dataset, nb_tasks=1):
        first_item = task_dataset[0]
        x, t, y = first_item
        assert x in observation_space
        assert y in reward_space


@pytest.mark.parametrize("dataset_name", ["mnist"])
def test_setting_obs_space_changes_when_transforms_change(dataset_name: str):
    """ TODO: Test that the `observation_space` property on the
    ClassIncrementalSetting reflects the data produced by the dataloaders, and
    that changing a transform on a Setting also changes the value of that
    property on both the Setting itself, as well as on the corresponding
    dataloaders/environments.
    """
    # dataset = ClassIncrementalSetting.available_datasets[dataset_name]
    setting = ClassIncrementalSetting(
        dataset=dataset_name,
        nb_tasks=1,
        transforms=[],
        train_transforms=[],
        val_transforms=[],
        test_transforms=[],
        batch_size=None,
        num_workers=0,
    )
    assert setting.observation_space.x == base_observation_spaces[dataset_name]
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
    from gym.vector.utils import batch_space
    assert train_env.observation_space == setting.observation_space

    reset_obs = train_env.reset()
    assert reset_obs[0] in train_env.observation_space[0], reset_obs[0].shape
    assert reset_obs[1] in train_env.observation_space[1]
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
    ##
    ##  ---------- Same tests for the val_environment --------------
    ##
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
    ##
    ##  ---------- Same tests for the test_environment --------------
    ##

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
def test_render():
    setting = ClassIncrementalSetting(dataset="mnist")
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