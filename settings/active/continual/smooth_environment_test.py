from collections import OrderedDict
from typing import Dict, List

import gym
import matplotlib.pyplot as plt
import numpy as np
import pytest

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
    assert env.gravity == starting_gravity
    assert env.length == starting_length
    env = gym.wrappers.Monitor(env, f"recordings/smooth_{environment_name}", force=True)
    env.seed(123)
    env.reset()

    assert env.gravity == starting_gravity
    assert env.length == starting_length
    plt.ion()

    params: Dict[int, Dict[str, float]] = OrderedDict()

    for step in range(total_steps):
        expected_steps = starting_length + (step / total_steps) * (end_length - starting_length)
        expected_gravity = starting_gravity + (step / total_steps) * (end_gravity - starting_gravity)

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
    plt.ioff()
    plt.close()



if __name__ == "__main__":
    test_monitor_env("CartPole-v0")
