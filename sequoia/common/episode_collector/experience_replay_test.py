from typing import Tuple
import gym

from sequoia.common.episode_collector.episode import Episode, Transition
from sequoia.common.gym_wrappers.convert_tensors import ConvertToFromTensors
from .experience_replay import ExperienceReplayLoader
from gym import spaces
from sequoia.common.typed_gym import _Env, _Space
import numpy as np


class SimpleEnv(gym.Env, _Env[int, int, float]):
    def __init__(self, target: int = 500, start_state: int = 0) -> None:
        super().__init__()
        self.observation_space = spaces.Discrete(1000)
        self.action_space = spaces.Discrete(2)
        self.target = target
        self.start_state = start_state
        self.state = start_state

    def reset(self) -> int:
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        assert action in self.action_space
        self.state += action
        return self.state, -abs(self.state - self.target), self.state == self.target, {}


def test_simple_env():
    env = SimpleEnv()
    obs = env.reset()
    assert obs == 0
    total_steps = 0
    done = False
    while not done:
        obs, reward, done, info = env.step(1)
        total_steps += 1
        assert obs == total_steps

        if obs == 500:
            assert done


def test_replay_buffer_with_simple_env():
    env = SimpleEnv()
    from .episode_collector import EpisodeCollector
    # from .replay_buffer import ReplayBuffer, EnvDataLoader
    
    def policy(obs: int, action_space: _Space[int]) -> int:
        return 1
    
    max_episodes = 2
    
    generator = EpisodeCollector(env=env, policy=policy, max_episodes=max_episodes)
    i = 0
    episode: Episode
    for i, episode in enumerate(generator):
        assert len(episode) == 500
        assert np.array_equal(episode.observations, np.arange(500))
        assert np.array_equal(episode.rewards, - np.arange(500)[::-1])
        assert np.array_equal(episode.actions, np.ones(500))
        assert episode.last_observation == 500
        assert i < max_episodes
    assert i == max_episodes - 1

    assert episode[499] == Transition(observation=499, action=1, reward=0, next_observation=500, info={}, done=True)


def test_experience_replay_simple():
    env = SimpleEnv()
    seed = 123
    batch_size = 1
    buffer_size = 100

    def policy(obs: int, action_space: _Space[int]) -> int:
        return 1
        # return action_space.sample()
    # TODO: First, add tests for the env dataset / dataloader / experience replay with envs that
    # have typed objects (e.g.) Observation/Action/Reward, tensors, etc.
    
    loader = ExperienceReplayLoader(
        env,
        batch_size=batch_size,
        buffer_size=100,
        max_episodes=10,
        policy=policy,
        seed=seed,
    )
    for i, batch in enumerate(loader):
        print(batch.done)
        
        if any(batch.done):
            assert False, batch
        assert isinstance(batch, Transition)
        assert batch in loader.item_space
        assert isinstance(batch.observation, np.ndarray)
        assert batch.shapes == {
            "action": (batch_size,),
            "done": (batch_size,),
            "info": None,
            "next_observation": (batch_size,),
            "observation": (batch_size,),
            "reward": (batch_size,),
        }
        assert batch.dtypes == {
            "action": np.dtype("int64"),
            "done": np.dtype("bool"),
            "info": None,
            "next_observation": np.dtype("int64"),
            "observation": np.dtype("int64"),
            "reward": np.dtype("float64"),
        }


import torch
from torch import Tensor


def test_experience_replay_with_tensor_env():
    env = SimpleEnv()
    from sequoia.common.gym_wrappers.convert_tensors import ConvertToFromTensors
    from sequoia.common.spaces.tensor_spaces import TensorDiscrete, TensorBox

    device = "cpu"

    env = ConvertToFromTensors(env, device=device)
    
    seed = 123
    batch_size = 10
    buffer_size = 100
    max_episodes = 2

    def policy(obs: int, action_space: _Space[int]) -> int:
        return action_space.sample()
        return 1

    loader = ExperienceReplayLoader(
        env,
        batch_size=batch_size,
        buffer_size=buffer_size,
        max_episodes=max_episodes,
        policy=policy,
        seed=seed,
    )
    for i, batch in enumerate(loader):
        assert isinstance(batch, Transition)
        assert batch in loader.item_space
        assert isinstance(batch.observation, Tensor)
        print(batch.done)
        
        assert batch.shapes == {
            "action": (10,),
            "done": (10,),
            "info": None,
            "next_observation": (10,),
            "observation": (10,),
            "reward": (10,),
        }
        assert batch.dtypes == {
            "action": torch.int64,
            "done": np.dtype(np.bool),
            "info": None,
            "next_observation": torch.int64,
            "observation": torch.int64,
            "reward": torch.float32,
        }


def test_with_typed_objects_and_tensors():
    # TODO: First, add tests for the env dataset / dataloader / experience replay with envs that
    # have typed objects (e.g.) Observation/Action/Reward, tensors, etc.
    env = gym.make("CartPole-v0")
    from sequoia.methods.models.base_model.rl.base_model_rl import UseObjectsWrapper
    env = UseObjectsWrapper(env)
    env = ConvertToFromTensors(env, device="cpu")
    
    from .experience_replay import ExperienceReplayLoader
    # from .replay_buffer import ReplayBuffer, EnvDataLoader
    
    def policy(obs: int, action_space: _Space[int]) -> int:
        assert False, obs
        return action_space.sample()

    max_episodes = 2
    loader = ExperienceReplayLoader(env=env, batch_size=3, policy=policy, max_episodes=max_episodes)
    i = 0
    for i, transitions in enumerate(loader):
        assert False, transitions
        # assert False, episode[len(episode)-1]

    assert i == max_episodes - 1
