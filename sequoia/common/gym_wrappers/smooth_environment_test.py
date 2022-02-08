from typing import Dict

import gym
import matplotlib.pyplot as plt
import numpy as np

from .smooth_environment import SmoothTransitions


def test_task_schedule():
    environment_name = "CartPole-v0"
    # wandb.init(name="SSCL/RL_testing/smooth", monitor_gym=True)
    original = gym.make(environment_name)
    starting_length = original.length
    starting_gravity = original.gravity

    end_length = 5 * starting_length
    end_gravity = 5 * starting_gravity
    total_steps = 100
    # Increase the length linearly up to 3 times the starting value.
    # Increase the gravity linearly up to 5 times the starting value.
    task_schedule: Dict[int, Dict[str, float]] = {
        # 0: dict(length=starting_length, gravity=starting_gravity),
        total_steps: dict(length=end_length, gravity=end_gravity),
    }
    env = SmoothTransitions(
        original,
        task_schedule=task_schedule,
    )
    # env = gym.wrappers.Monitor(env, f"recordings/smooth_{environment_name}", force=True)
    env.seed(123)
    env.reset()

    assert env.gravity == starting_gravity
    assert env.length == starting_length
    # plt.ion()

    params: Dict[int, Dict[str, float]] = {}

    for step in range(total_steps):
        expected_steps = starting_length + (step / total_steps) * (end_length - starting_length)
        expected_gravity = starting_gravity + (step / total_steps) * (
            end_gravity - starting_gravity
        )

        _, reward, done, _ = env.step(env.action_space.sample())
        assert np.isclose(env.length, expected_steps)
        assert np.isclose(env.gravity, expected_gravity)

        env.render()
        if done:
            env.reset()

        params[step] = env.current_task.copy()

        # print(f"New task: {env.current_task_dict()}")

    # assert False, params[step]
    env.close()
    # plt.ioff()
    plt.close()


def test_update_only_on_reset():
    """Test that when using the 'only_update_on_episode_end' argument with a
    value of True, the smooth updates don't occur during the episodes, but only
    once after an episode has ended (when `reset()` is called).
    """
    total_steps = 100
    original = gym.make("CartPole-v0")
    start_length = original.length
    end_length = 10.0
    task_schedule = {total_steps: dict(length=end_length)}
    env = SmoothTransitions(
        original,
        task_schedule=task_schedule,
        only_update_on_episode_end=True,
    )
    env.reset()
    env.seed(123)
    expected_length = start_length
    for i in range(total_steps):
        assert env.steps == i
        _, _, done, _ = env.step(env.action_space.sample())
        assert env.steps == i + 1
        if done:
            _ = env.reset()
            expected_length = start_length + ((i + 1) / total_steps) * (end_length - start_length)
        assert np.isclose(env.length, expected_length)


def test_task_id_is_always_None():
    total_steps = 100
    original = gym.make("CartPole-v0")
    start_length = original.length
    end_length = 10.0
    task_schedule = {total_steps: dict(length=end_length)}
    env = SmoothTransitions(
        original,
        task_schedule=task_schedule,
        only_update_on_episode_end=True,
        add_task_id_to_obs=True,
        add_task_dict_to_info=True,
    )

    for observation in (env.observation_space.sample() for i in range(100)):
        x, task_id = observation["x"], observation["task_labels"]
        assert task_id is None

    env.reset()
    env.seed(123)
    expected_length = start_length
    for i in range(total_steps):
        assert env.steps == i
        obs, _, done, _ = env.step(env.action_space.sample())

        x, task_id = obs["x"], obs["task_labels"]
        assert task_id is None

        assert env.steps == i + 1
        if done:
            obs = env.reset()
            x, task_id = obs["x"], obs["task_labels"]
            assert task_id is None

            expected_length = start_length + ((i + 1) / total_steps) * (end_length - start_length)
        assert np.isclose(env.length, expected_length)
