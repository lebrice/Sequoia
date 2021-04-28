import pytest
from .env_proxy import EnvironmentProxy
import psutil
from typing import ClassVar, Type
from sequoia.settings.active import IncrementalRLSetting
from sequoia.common.gym_wrappers.env_dataset import EnvDataset
from sequoia.common.gym_wrappers.env_dataset_test import TestEnvDataset
from functools import partial
import gym


class ProxyEnvDataset(EnvironmentProxy):
    def __init__(self, env, *args, **kwargs):
        env_fn = partial(EnvDataset, env=env, **kwargs)
        super().__init__(env_fn, setting_type=IncrementalRLSetting)



def test_sanity_check():
    env = ProxyEnvDataset(gym.make("CartPole-v0"))
    assert issubclass(type(env), EnvironmentProxy)


class TestProxyEnvDataset(TestEnvDataset):
    # IDEA: Reuse the tests for the EnvDataset, but using a proxy to the environment
    # instead.
    EnvDataset: ClassVar[Type[EnvDataset]] = ProxyEnvDataset


# from sequoia.common.gym_wrappers.
# @pytest.mark.xfail(reason="TODO")
# def test_env_proxy_has_same_len_as_env():
#     pass
# @pytest.mark.xfail(reason="TODO")
# def test_env_proxy_is_iterable():
#     pass