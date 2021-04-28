from functools import partial
from typing import ClassVar, Iterable, Tuple, Type, TypeVar

import gym
import pytest
from sequoia.common.gym_wrappers.env_dataset import EnvDataset
from sequoia.common.gym_wrappers.env_dataset_test import \
    TestEnvDataset as _TestEnvDataset
from sequoia.settings.active import IncrementalRLSetting
from sequoia.settings.passive.passive_environment import PassiveEnvironment
from sequoia.settings.passive.passive_environment_test import \
    TestPassiveEnvironment as _TestPassiveEnvironment
from sequoia.settings.active.continual.gym_dataloader import GymDataLoader
from sequoia.settings.active.continual.gym_dataloader_test import TestGymDataLoader as _TestGymDataLoader
from torch import Tensor
from sequoia.settings.assumptions import IncrementalSetting
from .env_proxy import EnvironmentProxy

# Note: import with underscores so we don't re-run those tests again.

EnvType = TypeVar("EnvType", bound=gym.Env, covariant=True)


def wrap_type_with_proxy(env_type: Type[EnvType]) -> EnvType:
    class _EnvProxy(EnvironmentProxy):
        def __init__(self, *args, **kwargs):
            env_fn = partial(env_type, *args, **kwargs)
            super().__init__(env_fn, setting_type=IncrementalSetting)
    return _EnvProxy


ProxyEnvDataset = wrap_type_with_proxy(EnvDataset)
ProxyPassiveEnvironment = wrap_type_with_proxy(PassiveEnvironment)
ProxyGymDataLoader = wrap_type_with_proxy(GymDataLoader)


class TestEnvironmentProxy(_TestEnvDataset, _TestPassiveEnvironment, _TestGymDataLoader):
    # IDEA: Reuse the tests for the EnvDataset, but using a proxy to the environment
    # instead.
    EnvDataset: ClassVar[Type[EnvDataset]] = ProxyEnvDataset

    # IDEA: Reuse the tests for the PassiveEnvironment, but using a proxy to the env.
    PassiveEnvironment: ClassVar[Type[PassiveEnvironment]] = ProxyPassiveEnvironment

    # Reuse the tests for the Gym DataLoader, using a proxy to the loader instead.
    GymDataLoader: ClassVar[Type[GymDataLoader]] = ProxyGymDataLoader


def test_sanity_check():
    env = ProxyEnvDataset(gym.make("CartPole-v0"))
    assert isinstance(env, EnvironmentProxy)
    assert issubclass(type(env), EnvironmentProxy)


from sequoia.common.gym_wrappers.utils import is_proxy_to


@pytest.mark.parametrize("use_wrapper", [False, True])
def test_is_proxy_to(use_wrapper: bool):
    import numpy as np
    from sequoia.common.transforms import Compose, Transforms

    transforms = Compose([Transforms.to_tensor, Transforms.three_channels])
    from sequoia.common.spaces import Image
    from torchvision.datasets import MNIST

    batch_size = 32
    dataset = MNIST("data", transform=transforms)
    obs_space = Image(0, 255, (1, 28, 28), np.uint8)
    obs_space = transforms(obs_space)

    env_type = ProxyPassiveEnvironment if use_wrapper else PassiveEnvironment
    env: Iterable[Tuple[Tensor, Tensor]] = env_type(
        dataset, batch_size=batch_size, n_classes=10, observation_space=obs_space,
    )
    if use_wrapper:
        assert isinstance(env, EnvironmentProxy)
        assert issubclass(type(env), EnvironmentProxy)
        assert is_proxy_to(env, PassiveEnvironment)
    else:
        assert not is_proxy_to(env, PassiveEnvironment)


# TODO: Write a test that first reproduces issue #204 and then check that removing
# `self.__environment.reset()` from __iter__ fixed it.
def test_issue_204():
    import psutil


def test_interaction_with_test_environment():
    # IDEA: Maybe write tests for the 'test' environments, and see that they work even
    # through the proxy?
    pass