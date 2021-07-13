import platform
from functools import partial
from typing import ClassVar, Iterable, List, Tuple, Type, TypeVar
from sequoia.common.gym_wrappers.utils import is_proxy_to

import gym
import numpy as np
import psutil
import pytest
from sequoia.common.gym_wrappers.env_dataset import EnvDataset
from sequoia.common.gym_wrappers.env_dataset_test import \
    TestEnvDataset as _TestEnvDataset
from sequoia.common.spaces import Image
from sequoia.common.transforms import Compose, Transforms
from sequoia.settings.rl import IncrementalRLSetting
from sequoia.settings.rl.continual.environment import GymDataLoader
from sequoia.settings.rl.continual.environment_test import \
    TestGymDataLoader as _TestGymDataLoader
from sequoia.settings.assumptions import IncrementalAssumption
from sequoia.settings.sl.environment import PassiveEnvironment
from sequoia.settings.sl.environment_test import \
    TestPassiveEnvironment as _TestPassiveEnvironment
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from .env_proxy import EnvironmentProxy

# Note: import with underscores so we don't re-run those tests again.

EnvType = TypeVar("EnvType", bound=gym.Env, covariant=True)


def wrap_type_with_proxy(env_type: Type[EnvType]) -> EnvType:
    class _EnvProxy(EnvironmentProxy):
        def __init__(self, *args, **kwargs):
            env_fn = partial(env_type, *args, **kwargs)
            super().__init__(env_fn, setting_type=IncrementalAssumption)

    return _EnvProxy


ProxyEnvDataset = wrap_type_with_proxy(EnvDataset)
ProxyPassiveEnvironment = wrap_type_with_proxy(PassiveEnvironment)
ProxyGymDataLoader = wrap_type_with_proxy(GymDataLoader)


class TestEnvironmentProxy(
    _TestEnvDataset, _TestPassiveEnvironment, _TestGymDataLoader
):
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


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="Not sure this would work the same on non-Linux systems.",
)
def test_issue_204():
    """ Test that reproduces the issue #204, which was that some zombie processes
    appeared to be created when iterating using an EnvironmentProxy.
    
    The issue appears to have been caused by calling `self.__environment.reset()` in
    `__iter__`, which I think caused another dataloader iterator to be created?
    """
    transforms = Compose([Transforms.to_tensor, Transforms.three_channels])

    batch_size = 2048
    num_workers = 12

    dataset = MNIST("data", transform=transforms)
    obs_space = Image(0, 255, (1, 28, 28), np.uint8)
    obs_space = transforms(obs_space)

    current_process = psutil.Process()
    print(
        f"Current process is using {current_process.num_threads()} threads, with "
        f" {len(current_process.children(recursive=True))} child processes."
    )
    starting_threads = current_process.num_threads()
    starting_processes = len(current_process.children(recursive=True))

    for use_wrapper in [False, True]:

        threads = current_process.num_threads()
        processes = len(current_process.children(recursive=True))
        assert threads == starting_threads
        assert processes == starting_processes

        env_type = ProxyPassiveEnvironment if use_wrapper else PassiveEnvironment
        env: Iterable[Tuple[Tensor, Tensor]] = env_type(
            dataset,
            batch_size=batch_size,
            n_classes=10,
            observation_space=obs_space,
            num_workers=num_workers,
            persistent_workers=True,
        )
        for i, _ in enumerate(env):
            threads = current_process.num_threads()
            processes = len(current_process.children(recursive=True))
            assert threads == starting_threads + num_workers
            assert processes == starting_processes + num_workers
            print(
                f"Current process is using {threads} threads, with "
                f" {processes} child processes."
            )

        for i, _ in enumerate(env):
            threads = current_process.num_threads()
            processes = len(current_process.children(recursive=True))
            assert threads == starting_threads + num_workers
            assert processes == starting_processes + num_workers
            print(
                f"Current process is using {threads} threads, with "
                f" {processes} child processes."
            )

        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            
            # env.render(mode="human")

            threads = current_process.num_threads()
            processes = len(current_process.children(recursive=True))
            if not done:
                assert threads == starting_threads + num_workers
                assert processes == starting_processes + num_workers
                print(
                    f"Current process is using {threads} threads, with "
                    f" {processes} child processes."
                )

        env.close()

        import time
        # Need to give it a second (or so) to cleanup.
        time.sleep(1)

        threads = current_process.num_threads()
        processes = len(current_process.children(recursive=True))
        assert threads == starting_threads
        assert processes == starting_processes


def test_interaction_with_test_environment():
    # IDEA: Maybe write tests for the 'test' environments, and see that they work even
    # through the proxy?
    pass
