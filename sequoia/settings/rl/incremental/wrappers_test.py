import gym
from .wrappers import RoundRobinWrapper, ConcatEnvsWrapper, RandomMultiEnvWrapper
from gym.wrappers import TimeLimit
from sequoia.common.gym_wrappers.episode_limit import EpisodeLimit
from sequoia.common.gym_wrappers.action_limit import ActionLimit
from typing import List


def test_concat():
    envs = [
        EpisodeLimit(TimeLimit(gym.make("CartPole-v0"), max_episode_steps=10), 5)
        for i in range(5)
    ]
    env = ConcatEnvsWrapper(envs)
    assert env.nb_tasks == 5

    for episode in range(25):
        print(f"Episode: {episode}")
        obs = env.reset()
        env_id = episode // 5
        assert env._current_task_id == env_id, episode
        step = 0
        done = False
        while not done:
            print(step)
            obs, rewards, done, info = env.step(env.action_space.sample())
            step += 1
            if step == 10:
                assert done
            assert step <= 10

    assert env.is_closed()

    # TODO: Test the same with StepLimit (ActionLimit) (which isn't stable atm because
    # it depends on each episode being 10 long, and CartPole ends earlier sometimes.)
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
    
    


def test_roundrobin():
    envs = [
        EpisodeLimit(TimeLimit(gym.make("CartPole-v0"), max_episode_steps=10), 5)
        for i in range(5)
    ]
    env = RoundRobinWrapper(envs)
    assert env.nb_tasks == 5

    for episode in range(25):
        print(f"Episode: {episode}")
        obs = env.reset()
        env_id = episode % 5
        assert env._current_task_id == env_id, episode
        step = 0
        done = False
        while not done:
            print(step)
            obs, rewards, done, info = env.step(env.action_space.sample())
            step += 1
            if step == 10:
                assert done
            assert step <= 10

    assert env.is_closed()


def test_random():
    envs = [
        EpisodeLimit(
            TimeLimit(gym.make("CartPole-v0"), max_episode_steps=10), max_episodes=5
        )
        for i in range(5)
    ]
    env = RandomMultiEnvWrapper(envs)
    env.seed(123)
    assert env.nb_tasks == 5
    task_ids: List[int] = []
    for episode in range(25):
        print(f"Episode: {episode}")
        obs = env.reset()
        env_id = episode // 5
        task_ids.append(env._current_task_id)
        step = 0
        done = False
        print(env._envs_is_closed)
        while not done:
            print(step)
            obs, rewards, done, info = env.step(env.action_space.sample())
            step += 1
            if step == 10:
                assert done
            assert step <= 10
    assert env.is_closed()
    from collections import Counter
    assert task_ids != list(range(5)) * 5
    assert Counter(task_ids) == {i: 5 for i in range(5)}
