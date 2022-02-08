""" TODO: Tests for the TestEnvironment of the ContinualSLSetting. """

from pathlib import Path
from typing import ClassVar, Type

import gym
import numpy as np
import pytest
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from sequoia.common.config import Config
from sequoia.common.metrics import ClassificationMetrics
from sequoia.common.spaces import Image
from sequoia.common.transforms import Compose, Transforms
from sequoia.settings.sl.environment import PassiveEnvironment

from .environment import ContinualSLEnvironment, ContinualSLTestEnvironment
from .results import ContinualSLResults


class TestContinualSLTestEnvironment:
    Environment: ClassVar[Type[Environment]] = ContinualSLEnvironment
    TestEnvironment: ClassVar[Type[TestEnvironment]] = ContinualSLTestEnvironment

    @pytest.fixture()
    def base_env(self):
        batch_size = 5
        transforms = Compose([Transforms.to_tensor, Transforms.three_channels])
        dataset = MNIST(
            "data", transform=Compose([Transforms.to_tensor, Transforms.three_channels])
        )
        max_samples = 100
        dataset = Subset(dataset, list(range(max_samples)))

        obs_space = Image(0, 255, (1, 28, 28), np.uint8)
        obs_space = transforms(obs_space)
        env = self.Environment(
            dataset,
            n_classes=10,
            batch_size=batch_size,
            observation_space=obs_space,
            pretend_to_be_active=True,
            drop_last=False,
        )
        assert env.observation_space == Image(0, 1, (batch_size, 3, 28, 28))
        assert env.action_space.shape == (batch_size,)
        assert env.reward_space == env.action_space
        return env

    @pytest.mark.parametrize("no_rewards", [True, False])
    def test_iteration_produces_results(
        self,
        no_rewards: bool,
        base_env: ContinualSLEnvironment,
        tmp_path: Path,
        config: Config,
    ):
        """TODO: Test that when iterating through the env as a dataloader and sending
        actions produces results.
        """
        env = self.TestEnvironment(
            base_env,
            directory=tmp_path,
            step_limit=100 // base_env.batch_size,
            no_rewards=no_rewards,
        )
        env.config = config

        for obs, rewards in env:
            assert rewards is None
            action = env.action_space.sample()
            rewards = env.send(action)
            assert (rewards is None) == env.no_rewards

        assert env.is_closed()
        results = env.get_results()
        self.validate_results(results)

    def validate_results(self, results: ContinualSLResults):
        assert isinstance(results, ContinualSLResults)
        assert isinstance(results.average_metrics, ClassificationMetrics)
        assert results.objective > 0
        # TODO: Fix this problem:
        assert results.average_metrics.n_samples in [95, 100]

    @pytest.mark.parametrize("no_rewards", [True, False])
    def test_gym_interaction_produces_results(
        self, no_rewards: bool, base_env: PassiveEnvironment, tmp_path: Path, config: Config
    ):
        """TODO: Test that when iterating through the env as a dataloader and sending
        actions produces results.
        """
        env = self.TestEnvironment(
            base_env,
            directory=tmp_path,
            step_limit=100 // base_env.batch_size,
            no_rewards=no_rewards,
        )
        env.config = config
        done = False
        obs = env.reset()
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, rewards, done, info = env.step(action)
            steps += 1
            assert (rewards is None) == env.no_rewards

            if steps > 20:
                pytest.fail("Shouldn't have gone longer than 20 steps!")

        # BUG: There's currently a weird off-by-1 error with the total number of steps,
        # which makes these checks for `is_closed()` fail. However, in practice we don't
        # try to iterate twice on the env, so it's not a big deal.
        # FIXME: Fix this check:
        assert env.is_closed()
        # FIXME: Fix this check:
        with pytest.raises((gym.error.ClosedEnvironmentError, gym.error.Error)):
            env.reset()
        # FIXME: Fix this check:
        with pytest.raises(gym.error.ClosedEnvironmentError):
            _ = env.step(env.action_space.sample())

        results = env.get_results()
        self.validate_results(results)
