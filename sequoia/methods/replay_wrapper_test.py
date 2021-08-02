from typing import Iterable, Tuple

import numpy as np
import pytest
import torch
from sequoia.common.spaces import TypedDictSpace, TensorDiscrete, TensorBox
from sequoia.common.spaces.image import Image, ImageTensorSpace
from sequoia.common.transforms import Compose, Transforms
from sequoia.settings.sl import SLEnvironment
from sequoia.settings.sl.continual import Observations, Actions, Rewards
from torch import Tensor
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from gym import spaces
from .replay_wrapper import ReplayEnvWrapper, Buffer
from gym.vector.utils.spaces import batch_space


class TestReplayWrapper:
    def sl_env(
        self,
        task_id: int = 0,
        n_samples: int = 100,
        batch_size: int = 10,
        pretend_to_be_active: bool = False,
    ):
        x_shape = (1,)
        dataset = TensorDataset(
            torch.ones([n_samples, *x_shape], dtype=torch.uint8) * task_id,  # x
            torch.ones((n_samples,), dtype=torch.long) * task_id,  # task labels
            torch.ones((n_samples,), dtype=torch.long) * task_id,  # y
        )

        def split_batch_fn(batch) -> Tuple[Observations, Rewards]:
            x, t, y = batch
            return (
                Observations(x=x, task_labels=t),
                Rewards(y=y),
            )

        env: Iterable[Tuple[Tensor, Tensor]] = SLEnvironment(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            split_batch_fn=split_batch_fn,
            observation_space=TypedDictSpace(
                x=TensorBox(0, 1, x_shape, dtype=torch.uint8),
                task_labels=TensorDiscrete(5),
                dtype=Observations,
            ),
            action_space=TypedDictSpace(y_pred=TensorDiscrete(10), dtype=Actions,),
            reward_space=TypedDictSpace(y=TensorDiscrete(10), dtype=Rewards,),
            pretend_to_be_active=pretend_to_be_active,
        )
        return env

    @pytest.mark.parametrize("seed", list(range(10)))
    @pytest.mark.parametrize("pretend_to_be_active", [False, True])
    def test_works_with_step(self, seed: int, pretend_to_be_active: bool):
        batch_size = 10
        sample_size = 5
        n_samples = 100
        # Create the envs to be wrapped for each task.
        first_env = self.sl_env(
            task_id=0,
            batch_size=batch_size,
            n_samples=n_samples,
            pretend_to_be_active=pretend_to_be_active,
        )
        second_env = self.sl_env(
            task_id=1,
            batch_size=batch_size,
            n_samples=n_samples,
            pretend_to_be_active=pretend_to_be_active,
        )

        env = wrapper = ReplayEnvWrapper(
            first_env, capacity=500, sample_size=sample_size, task_id=0
        )
        env.seed(seed)
        env.enable_collection()
        assert not env.sampling_enabled

        obs = env.reset()
        # Check that the observation from the env hasn't yet been pushed to the buffer,
        # since we don't yet have the rewards.
        assert len(env.buffer) == 0

        obs, rewards, done, info = env.step(env.action_space.sample())
        # Check that the observation and reward from the env has been pushed into the buffer:
        assert len(env.buffer) == batch_size

        # Check that the observation doesn't have any sampled values:
        assert obs.x.shape[0] == batch_size

        env = wrapper = wrapper.for_next_env(second_env, task_id=1)
        env.seed(seed)
        assert wrapper.sampling_enabled
        assert wrapper.collection_enabled

        # The observation space should reflect the batch_size + sample_size!
        # (Same goes for the action and reward spaces).
        assert env.observation_space == batch_space(
            env.single_observation_space, batch_size + sample_size
        )
        assert env.action_space == batch_space(
            env.single_action_space, batch_size + sample_size
        )
        assert env.reward_space == batch_space(
            env.single_reward_space, batch_size + sample_size
        )

        obs = env.reset()

        # Check that the buffer still contains only one bach (from the first env)
        assert len(env.buffer) == batch_size

        # The first `batch_size` task labels should come from the buffer, and the last
        # `sample_size` from the environment.
        assert obs.task_labels.tolist() == [1] * batch_size + [0] * sample_size

        obs, rewards, done, info = env.step(env.action_space.sample())
        # Check that the buffer now contains two batches: One from the first env, and one
        # from the second.
        assert len(env.buffer) == batch_size * 2

        # NOTE: Since we gave a task-id when creating the second env (and passed it to
        # `for_next_env`), the new wrapper will not sample items from the current task.

        # The first `batch_size` rewards should come from the buffer, and the last
        # `sample_size` from the environment.
        assert rewards.y.tolist() == [1] * batch_size + [0] * sample_size
        # The first `batch_size` task labels should come from the buffer, and the last
        # `sample_size` from the environment.
        assert obs.task_labels.tolist() == [1] * batch_size + [0] * sample_size

    @pytest.mark.parametrize("seed", list(range(10)))
    @pytest.mark.parametrize("pretend_to_be_active", [False, True])
    def test_works_with_iter(self, seed: int, pretend_to_be_active: bool):
        batch_size = 10
        sample_size = 5
        n_samples = 100
        # Create the envs to be wrapped for each task.
        first_env = self.sl_env(
            task_id=0,
            batch_size=batch_size,
            n_samples=n_samples,
            pretend_to_be_active=pretend_to_be_active,
        )
        second_env = self.sl_env(
            task_id=1,
            batch_size=batch_size,
            n_samples=n_samples,
            pretend_to_be_active=pretend_to_be_active,
        )
        env = wrapper = ReplayEnvWrapper(
            first_env, capacity=500, sample_size=sample_size, task_id=0
        )
        env.enable_collection()
        assert not env.sampling_enabled

        for obs, rewards in env:
            # Check that the observation doesn't have any sampled values:
            assert obs.x.shape[0] == batch_size
            assert (rewards is None) == pretend_to_be_active
            if rewards is None:
                # Check that the observation from the env hasn't yet been pushed to the buffer,
                # since we don't yet have the rewards.
                assert len(env.buffer) == 0
                rewards = env.send(env.action_space.sample())
            # Check that the observation and reward from the env has been pushed into the buffer:
            assert len(env.buffer) == batch_size
            break

        assert len(env.buffer) == batch_size

        env = wrapper = wrapper.for_next_env(second_env, task_id=1)
        env.seed(seed)

        assert wrapper.sampling_enabled
        assert wrapper.collection_enabled

        # The observation space should reflect the batch_size + sample_size!
        # (Same goes for the action and reward spaces).
        assert env.observation_space == batch_space(
            env.single_observation_space, batch_size + sample_size
        )
        assert env.action_space == batch_space(
            env.single_action_space, batch_size + sample_size
        )
        assert env.reward_space == batch_space(
            env.single_reward_space, batch_size + sample_size
        )

        # env_iter = iter(env)
        # for obs in env_iter:
        for i, (obs, rewards) in enumerate(env):
            # Check that the observation doesn't have any sampled values:
            assert obs.x.shape[0] == batch_size + sample_size

            # The first `batch_size` task labels should come from the buffer, and the last
            # `sample_size` from the environment.
            assert obs.task_labels.tolist() == [1] * batch_size + [0] * sample_size
            assert (rewards is None) == pretend_to_be_active

            if rewards is None:
                assert len(env.buffer) == batch_size * (i + 1)
                rewards = env.send(env.action_space.sample())
                # Check that the observation and reward from the env has been pushed into the buffer:
            assert len(env.buffer) == batch_size * (i + 2)

            # The first `batch_size` rewards should come from the buffer, and the last
            # `sample_size` from the environment.
            # BUG: This is SOMETIMES wrong:
            assert rewards.y.tolist() == [1] * batch_size + [0] * sample_size

        expected_n_batches = (
            n_samples // batch_size
            + (1 if n_samples % batch_size and env.drop_last else 0)
        ) - 1
        assert i == expected_n_batches

