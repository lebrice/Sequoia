from typing import Type, ClassVar, Optional
import gym
import numpy as np
import pytest
import torch
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from sequoia.common.spaces.utils import batch_space
from sequoia.common.gym_wrappers import EnvDataset, PixelObservationWrapper
from sequoia.conftest import param_requires_atari_py
from sequoia.utils import take
from sequoia.utils.logging_utils import get_logger
from torch import Tensor

from .environment import GymDataLoader
from .make_env import make_batched_env

logger = get_logger(__file__)


class TestGymDataLoader:
    # Grouping tests into a class so we can inherit from it in another test module, for
    # instance in the tests for EnvironmentProxy class.
    GymDataLoader: ClassVar[Type[GymDataLoader]] = GymDataLoader

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize(
        "env_name", ["CartPole-v0", param_requires_atari_py("Breakout-v0")]
    )
    def test_spaces(self, env_name: str, batch_size: int):
        dataset = EnvDataset(make_batched_env(env_name, batch_size=batch_size))

        batched_obs_space = dataset.observation_space
        # NOTE: the VectorEnv class creates the 'batched' action space by creating a
        # Tuple of the single action space, of length 'N', which seems a bit weird.
        # batched_action_space = vector_env.action_space
        batched_action_space = batch_space(dataset.single_action_space, batch_size)

        dataloader_env = self.GymDataLoader(dataset, batch_size=batch_size)
        assert dataloader_env.observation_space == batched_obs_space
        assert dataloader_env.action_space == batched_action_space

        dataloader_env.reset()
        for observation_batch in take(dataloader_env, 3):
            if isinstance(observation_batch, Tensor):
                observation_batch = observation_batch.cpu().numpy()
            assert observation_batch in batched_obs_space

            actions = dataloader_env.action_space.sample()
            assert len(actions) == batch_size
            assert actions in batched_action_space

            rewards = dataloader_env.send(actions)
            # BUG: rewards has dtype np.float64, while the space has np.float32.
            assert len(rewards) == batch_size
            assert rewards in dataloader_env.reward_space

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    @pytest.mark.parametrize(
        "env_name", ["CartPole-v0", param_requires_atari_py("Breakout-v0")]
    )
    def test_max_steps_is_respected(self, env_name: str, batch_size: int):
        max_steps = 5
        env_name = "CartPole-v0"
        env = make_batched_env(env_name, batch_size=batch_size)
        dataset = EnvDataset(env)
        from sequoia.common.gym_wrappers.action_limit import ActionLimit
        dataset = ActionLimit(dataset, max_steps=max_steps * (batch_size or 1))
        env: GymDataLoader = self.GymDataLoader(dataset)
        env.reset()
        i = 0
        for i, obs in enumerate(env):
            assert obs in env.observation_space
            assert i < max_steps, f"Max steps should have been respected: {i}"
            env.send(env.action_space.sample())
        assert i == max_steps - 1
        env.close()

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    @pytest.mark.parametrize("seed", [None, 123, 456])
    # @pytest.mark.parametrize(
    #     "env_name", ["CartPole-v0", param_requires_atari_py("Breakout-v0")]
    # )
    def test_multiple_epochs_works(self, batch_size: Optional[int], seed: Optional[int]):
        epochs = 3
        max_steps_per_episode = 10
        from gym.wrappers import TimeLimit
        from sequoia.conftest import DummyEnvironment
        from sequoia.common.gym_wrappers import AddDoneToObservation
        def env_fn():
            # FIXME: Using the DummyEnvironment for now since it's easier to debug with.
            # env = gym.make(env_name)
            env = DummyEnvironment()
            env = AddDoneToObservation(env)
            env = TimeLimit(env, max_episode_steps=max_steps_per_episode)
            return env

        # assert False, [env_fn(i).unwrapped for i in range(4)]
        # env = gym.vector.make(env_name, num_envs=(batch_size or 1))
        env = make_batched_env(env_fn, batch_size=batch_size)
        
        
        batched_env = env
        # from sequoia.common.gym_wrappers.episode_limit import EpisodeLimit
        # env = EpisodeLimit(env, max_episodes=epochs)
        from sequoia.common.gym_wrappers.convert_tensors import ConvertToFromTensors
        env = ConvertToFromTensors(env)

        env = EnvDataset(env, max_steps_per_episode=max_steps_per_episode)

        env: GymDataLoader = self.GymDataLoader(env)
        # BUG: Seems to be a little bug in the shape of the items yielded by the env due
        # to the concat_fn of the DataLoader.
        # if batch_size and batch_size >= 1:
        #     assert False, (env.reset().shape, env.observation_space, next(iter(env)).shape)
        env.seed(seed)

        all_rewards = []
        with env:
            for epoch in range(epochs):
                for step, obs in enumerate(env):
                    print(f"'epoch' {epoch}, step {step}:, obs: {obs}")
                    assert obs in env.observation_space, obs.shape
                    assert (  # BUG: This isn't working: (sometimes!)
                        step < max_steps_per_episode
                    ), "Max steps per episode should have been respected."
                    rewards = env.send(env.action_space.sample())

                    if batch_size is None:
                        all_rewards.append(rewards)
                    else:
                        all_rewards.extend(rewards)

                # Since in the VectorEnv, 'episodes' are infinite, we must have
                # reached the limit of the number of steps, while in a single
                # environment, the episode might have been shorter.
                assert step <= max_steps_per_episode - 1

            assert epoch == epochs - 1

        if batch_size in [None, 1]:
            # Some episodes might last shorter than the max number of steps per episode,
            # therefore the total should be at most this much:
            assert len(all_rewards) <= epochs * max_steps_per_episode
        else:
            # The maximum number of steps per episode is set, but the env is vectorized,
            # so the number of 'total' rewards we get from all envs should be *exactly*
            # this much:
            assert len(all_rewards) == epochs * max_steps_per_episode * batch_size

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("env_name", [param_requires_atari_py("Breakout-v0")])
    def test_reward_isnt_always_one(self, env_name: str, batch_size: int):
        epochs = 3
        max_steps_per_episode = 100

        env = make_batched_env(env_name, batch_size=batch_size)
        dataset = EnvDataset(env, max_steps_per_episode=max_steps_per_episode)

        env: GymDataLoader = self.GymDataLoader(env=dataset)
        all_rewards = []
        with env:
            env.reset()
            for epoch in range(epochs):
                for i, batch in enumerate(env):
                    rewards = env.send(env.action_space.sample())
                    all_rewards.extend(rewards)

        assert all_rewards != np.ones(len(all_rewards)).tolist()

    @pytest.mark.parametrize("env_name", ["CartPole-v0"])
    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
    def test_batched_state(self, env_name: str, batch_size: int):
        max_steps_per_episode = 10

        env = make_batched_env(env_name, batch_size=batch_size)
        dataset = EnvDataset(env, max_steps_per_episode=max_steps_per_episode)

        env: GymDataLoader = GymDataLoader(
            dataset, batch_size=batch_size,
        )
        with gym.make(env_name) as temp_env:
            state_shape = temp_env.observation_space.shape
            action_shape = temp_env.action_space.shape

        state_shape = (batch_size, *state_shape)
        action_shape = (batch_size, *action_shape)
        reward_shape = (batch_size,)

        state = env.reset()
        assert state.shape == state_shape
        env.seed(123)
        i = 0
        for obs_batch in take(env, 5):
            assert obs_batch.shape == state_shape

            random_actions = env.action_space.sample()
            assert torch.as_tensor(random_actions).shape == action_shape
            assert temp_env.action_space.contains(random_actions[0])

            reward = env.send(random_actions)
            assert reward.shape == reward_shape
            i += 1
        assert i == 5

    @pytest.mark.parametrize("env_name", ["CartPole-v0"])
    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
    def test_batched_pixels(self, env_name: str, batch_size: int):
        max_steps_per_episode = 10

        wrappers = [PixelObservationWrapper]
        env = make_batched_env(env_name, wrappers=wrappers, batch_size=batch_size)
        dataset = EnvDataset(env, max_steps_per_episode=max_steps_per_episode)

        with gym.make(env_name) as temp_env:
            for wrapper in wrappers:
                temp_env = wrapper(temp_env)

            state_shape = temp_env.observation_space.shape
            action_shape = temp_env.action_space.shape

        state_shape = (batch_size, *state_shape)
        action_shape = (batch_size, *action_shape)
        reward_shape = (batch_size,)

        env = self.GymDataLoader(dataset, batch_size=batch_size,)
        assert isinstance(env.observation_space, spaces.Box)
        assert len(env.observation_space.shape) == 4
        assert env.observation_space.shape[0] == batch_size

        env.seed(1234)
        for i, batch in enumerate(env):
            assert len(batch) == batch_size

            if isinstance(batch, Tensor):
                batch = batch.cpu().numpy()
            assert batch in env.observation_space

            random_actions = env.action_space.sample()
            assert torch.as_tensor(random_actions).shape == action_shape
            assert temp_env.action_space.contains(random_actions[0])

            reward = env.send(random_actions)
            assert reward.shape == reward_shape
