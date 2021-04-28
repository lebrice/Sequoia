import pytest
from .env_proxy import EnvironmentProxy
import psutil
from typing import ClassVar, Type
from sequoia.settings.active import IncrementalRLSetting
from sequoia.common.gym_wrappers.env_dataset import EnvDataset
from sequoia.common.gym_wrappers.env_dataset_test import TestEnvDataset as _TestEnvDataset
from functools import partial
import gym

from sequoia.settings.passive.passive_environment import PassiveEnvironment
from sequoia.settings.passive.passive_environment_test import TestPassiveEnvironment as _TestPassiveEnvironment

# Note: import with underscores so we don't re-run those tests again.


class ProxyEnvDataset(EnvironmentProxy):
    def __init__(self, env, *args, **kwargs):
        env_fn = partial(EnvDataset, env=env, **kwargs)
        super().__init__(env_fn, setting_type=IncrementalRLSetting)


class ProxyPassiveEnvironment(EnvironmentProxy):
    def __init__(self, *args, **kwargs):
        env_fn = partial(PassiveEnvironment, *args, **kwargs)
        super().__init__(env_fn, setting_type=IncrementalRLSetting)


class TestEnvironmentProxy(_TestEnvDataset, _TestPassiveEnvironment):
    # IDEA: Reuse the tests for the EnvDataset, but using a proxy to the environment
    # instead.
    EnvDataset: ClassVar[Type[EnvDataset]] = ProxyEnvDataset

    # IDEA: Reuse the tests for the PassiveEnvironment, but using a proxy to the env.
    PassiveEnvironment: ClassVar[Type[PassiveEnvironment]] = ProxyPassiveEnvironment


def test_sanity_check():
    env = ProxyEnvDataset(gym.make("CartPole-v0"))
    assert isinstance(env, EnvironmentProxy)
    assert issubclass(type(env), EnvironmentProxy)


@pytest.mark.parametrize("use_wrapper", [False, True])
def test_proxy_isinstance_of_wrapped_type(use_wrapper: bool):
    from sequoia.common.transforms import Transforms, Compose
    import numpy as np
    transforms = Compose([Transforms.to_tensor, Transforms.three_channels])
    from torchvision.datasets import MNIST
    from sequoia.common.spaces import Image
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

    # NOTE: This is a bit funky!
    assert isinstance(env, PassiveEnvironment)



# from sequoia.common.gym_wrappers.
# @pytest.mark.xfail(reason="TODO")
# def test_env_proxy_has_same_len_as_env():
#     pass
# @pytest.mark.xfail(reason="TODO")
# def test_env_proxy_is_iterable():
#     pass