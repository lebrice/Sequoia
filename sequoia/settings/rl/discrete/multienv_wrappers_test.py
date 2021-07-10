import gym
from .multienv_wrappers import (
    RoundRobinWrapper,
    ConcatEnvsWrapper,
    RandomMultiEnvWrapper,
)
from gym.wrappers import TimeLimit
from sequoia.common.gym_wrappers.episode_limit import EpisodeLimit
from sequoia.common.gym_wrappers.action_limit import ActionLimit
from sequoia.common.spaces import TypedDictSpace
from sequoia.common.gym_wrappers.env_dataset import EnvDataset

from typing import List, Optional
import pytest
from gym import spaces

from sequoia.utils.utils import unique_consecutive, unique_consecutive_with_index


class TestMultiEnvWrappers:
    @pytest.fixture()
    def iterable_env(self) -> gym.Env:
        return EnvDataset(gym.make("CartPole-v0"))

    @pytest.mark.parametrize("add_task_ids", [False, True])
    @pytest.mark.parametrize("nb_tasks", [5, 1])
    def test_concat(self, add_task_ids: bool, nb_tasks: int):
        def set_attributes(env: gym.Env, **attributes) -> gym.Env:
            for k, v in attributes.items():
                setattr(env.unwrapped, k, v)
            return env

        max_episodes_per_task = 5
        envs = [
            EpisodeLimit(
                TimeLimit(
                    set_attributes(gym.make("CartPole-v0"), length=0.1 + 0.2 * i),
                    max_episode_steps=10,
                ),
                max_episodes=max_episodes_per_task,
            )
            for i in range(nb_tasks)
        ]

        env = ConcatEnvsWrapper(envs, add_task_ids=add_task_ids)
        assert env.nb_tasks == nb_tasks
        
        if add_task_ids:
            assert env.observation_space["task_labels"] == spaces.Discrete(env.nb_tasks)
        lengths = []
        for episode in range(nb_tasks * max_episodes_per_task):
            print(f"Episode: {episode}, length: {round(env.length, 5)}")
            obs = env.reset()
            lengths.append(env.length)

            env_id = episode // max_episodes_per_task
            assert env._current_task_id == env_id, episode
            if add_task_ids:
                assert obs["task_labels"] == env_id
            step = 0
            done = False
            while not done:
                obs, rewards, done, info = env.step(env.action_space.sample())
                step += 1
                if step == 10:
                    assert done
                assert step <= 10

        # NOTE: It's pretty cool that we actually recover something like the task
        # schedule here! :D
        episode_task_schedule = dict(unique_consecutive_with_index(lengths))
        assert episode_task_schedule == {   
            i * max_episodes_per_task: 0.1 + 0.2 * i for i in range(nb_tasks)
        }
        assert env.is_closed()

        # TODO: This does the same with an additional StepLimit (ActionLimit) wrapper,
        # and isn't stable because it depends on each episode being 10 long, and
        # CartPole ends earlier sometimes.
        # envs = [
        #     ActionLimit(TimeLimit(gym.make("CartPole-v0"), max_episode_steps=10), max_steps=50)
        #     for i in range(5)
        # ]
        # env = ConcatEnvsWrapper(envs)
        # assert env.nb_tasks == 5

        # for episode in range(25):
        #     print(f"Episode: {episode}")
        #     print(env.max_steps, env.step_count())
        #     obs = env.reset()
        #     env_id = episode // 5
        #     assert env._current_task_id == env_id, episode
        #     step = 0
        #     done = False
        #     while not done:
        #         print(step)
        #         obs, rewards, done, info = env.step(env.action_space.sample())
        #         step += 1
        #         if step == 10:
        #             assert done
        #         assert step <= 10

        # assert env.is_closed()

    @pytest.mark.parametrize("add_task_ids", [False, True])
    @pytest.mark.parametrize("nb_tasks", [5, 1])
    def test_roundrobin(self, add_task_ids: bool, nb_tasks: int):
        max_episodes_per_task = 5
        max_episode_steps = 10
        envs = [
            EpisodeLimit(
                TimeLimit(gym.make("CartPole-v0"), max_episode_steps=max_episode_steps),
                max_episodes=max_episodes_per_task,
            )
            for i in range(nb_tasks)
        ]
        env = RoundRobinWrapper(envs, add_task_ids=add_task_ids)
        assert env.nb_tasks == nb_tasks
        if add_task_ids:
            assert env.observation_space["task_labels"] == spaces.Discrete(env.nb_tasks)
        else:
            assert env.observation_space == env._envs[0].observation_space

        for episode in range(nb_tasks * max_episodes_per_task):
            print(f"Episode: {episode}")
            obs = env.reset()
            env_id = episode % nb_tasks
            assert env._current_task_id == env_id, episode
            step = 0
            done = False
            while not done:
                print(step)
                obs, rewards, done, info = env.step(env.action_space.sample())
                step += 1
                if step == max_episode_steps:
                    assert done
                assert step <= max_episode_steps

        assert env.is_closed()

    def test_random(self):
        episodes_per_task = 5
        max_episode_steps = 10
        nb_tasks = 5
        envs = [
            EpisodeLimit(
                TimeLimit(gym.make("CartPole-v0"), max_episode_steps=max_episode_steps),
                max_episodes=episodes_per_task,
            )
            for i in range(nb_tasks)
        ]
        env = RandomMultiEnvWrapper(envs)
        env.seed(123)
        assert env.nb_tasks == nb_tasks
        task_ids: List[int] = []
        for episode in range(nb_tasks * episodes_per_task):
            print(f"Episode: {episode}")
            obs = env.reset()
            env_id = episode // nb_tasks
            task_ids.append(env._current_task_id)
            step = 0
            done = False
            print(env._envs_is_closed)
            while not done:
                print(step)
                obs, rewards, done, info = env.step(env.action_space.sample())
                step += 1
                if step == max_episode_steps:
                    assert done
                assert step <= max_episode_steps
        assert env.is_closed()
        from collections import Counter

        # Assert that the task ids are 'random':
        import torch

        assert len(torch.unique_consecutive(torch.as_tensor(task_ids))) > nb_tasks
        assert Counter(task_ids) == {i: episodes_per_task for i in range(nb_tasks)}

    def test_iteration(self, iterable_env: gym.Env):
        """ TODO: Interesting bug! Might be because when switching between envs, we're
        setting the 'cached' attributes onto the unwrapped env, and so when we move to
        another env, we all of a sudden don't have those attributes!
        """
        max_episode_steps = 10
        episodes_per_task = 5
        add_task_ids = True
        nb_tasks = 5

        envs = [
            EpisodeLimit(
                TimeLimit(
                    iterable_env,
                    max_episode_steps=max_episode_steps,
                ),
                max_episodes=episodes_per_task,
            )
            for i in range(nb_tasks)
        ]

        on_task_switch_received_task_ids: List[Optional[int]] = []

        def on_task_switch(task_id: Optional[int]) -> None:
            print(f"On task switch: {task_id}.")
            on_task_switch_received_task_ids.append(task_id)

        env = ConcatEnvsWrapper(
            envs, add_task_ids=add_task_ids, on_task_switch_callback=on_task_switch
        )
        env.seed(123)
        assert env.nb_tasks == nb_tasks
        if add_task_ids:
            assert env.observation_space == TypedDictSpace(
                x=env._envs[0].observation_space, task_labels=spaces.Discrete(nb_tasks),
            )
        else:
            assert env.observation_space == env._envs[0].observation_space
        assert env.observation_space.sample() in env.observation_space
        task_ids: List[int] = []

        for episode in range(nb_tasks * episodes_per_task):
            env_id = episode // episodes_per_task

            episode_task_ids: List[int] = []
            for step, obs in enumerate(env):
                assert obs in env.observation_space
                print(f"Episode {episode}, Step {step}: obs: {obs}")

                if add_task_ids:
                    assert list(obs.keys()) == ["x", "task_labels"]
                    obs_task_id = obs["task_labels"]
                    episode_task_ids.append(obs_task_id)
                    print(f"obs Task id: {obs_task_id}")

                rewards = env.send(env.action_space.sample())
                if step > max_episode_steps:
                    assert False, "huh?"

            if add_task_ids:
                assert (
                    len(set(episode_task_ids)) == 1
                ), f"all observations within an episode should have the same task id.: {episode_task_ids}"
                # Add the unique task id from this episode to the list of all task ids.
                task_ids.extend(set(episode_task_ids))

        assert env.is_closed()
        from collections import Counter

        if add_task_ids:
            assert task_ids == sum(
                [[i] * episodes_per_task for i in range(nb_tasks)], []
            )
            # should have received one per boundary
            assert on_task_switch_received_task_ids == list(range(1, nb_tasks))
            assert Counter(task_ids) == {i: episodes_per_task for i in range(nb_tasks)}
        else:
            assert on_task_switch_received_task_ids == [None] * (nb_tasks - 1)

    def test_adding_envs(self):
        from sequoia.common.gym_wrappers.env_dataset import EnvDataset

        env_1 = EnvDataset(
            EpisodeLimit(
                TimeLimit(gym.make("CartPole-v1"), max_episode_steps=10), max_episodes=5
            )
        )
        env_2 = EnvDataset(
            EpisodeLimit(
                TimeLimit(gym.make("CartPole-v1"), max_episode_steps=10), max_episodes=5
            )
        )
        chained_env = env_1 + env_2
        assert chained_env._envs[0] is env_1
        assert chained_env._envs[1] is env_2
        # TODO: Do we add a 'len' attribute?
        # assert False, len(chained_env)
        # assert


def test_batched_envs():
    """ TODO: Not sure how this will work with batched envs, but if it did, we could
    allow batch_size > 1 in Discrete, or batched custom envs in Incremental.
    """
