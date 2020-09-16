from typing import Dict, List

import gym
import matplotlib.pyplot as plt
import pytest

from gym.envs.classic_control import CartPoleEnv
from .multi_task_environment import MultiTaskEnvironment
supported_environments: List[str] = ["CartPole-v0"]

def test_task_schedule():
    original: CartPoleEnv = gym.make("CartPole-v0")
    starting_length = original.length
    starting_gravity = original.gravity

    task_schedule = {
        10: dict(length=0.1),
        20: dict(length=0.2, gravity=-12.0),
        30: dict(gravity=0.9),
    }
    env = MultiTaskEnvironment(original, task_schedule=task_schedule)
    env.seed(123)
    env.reset()
    for step in range(100):
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
        if done:
            env.reset()

        if 0 <= step < 10:
            assert env.length == starting_length and env.gravity == starting_gravity
        elif 10 <= step < 20:
            assert env.length == 0.1
        elif 20 <= step < 30:
            assert env.length == 0.2 and env.gravity == -12.0
        elif step >= 30:
            assert env.length == starting_length and env.gravity == 0.9

    env.close()


@pytest.mark.parametrize("environment_name", supported_environments)
def test_multi_task(environment_name: str):   
    original = gym.make(environment_name)  
    env = MultiTaskEnvironment(original)
    env.reset()
    env.seed(123)
    plt.ion()
    default_task = env.default_task
    for task_id in range(5):
        for i in range(20):
            observation, reward, done, info = env.step(env.action_space.sample())
            env.render()
        env.reset(new_random_task=True)
        print(f"New task: {env.current_task}")
    env.close()
    plt.ioff()
    plt.close()



@pytest.mark.skip(reason="This generates some output, uncomment this to run it.")
@pytest.mark.parametrize("environment_name", supported_environments)
def test_monitor_env(environment_name):
    original = gym.make(environment_name)
    # original = CartPoleEnv()
    env = MultiTaskEnvironment(original)
    env = gym.wrappers.Monitor(
        env,
        f"recordings/multi_task_{environment_name}",
        force=True,
        write_upon_reset=False,
    )
    env.seed(123)
    env.reset()
    
    plt.ion()

    task_param_values: List[Dict] = []
    default_length: float = env.length
    from gym.wrappers import Monitor
    for task_id in range(20):
        for i in range(100):
            observation, reward, done, info = env.step(env.action_space.sample())
            env.render()
            if done:
                env.reset(new_task=False)

            task_param_values.append(env.current_task.copy())
            # env.update_task(length=(i + 1) / 100 * 2 * default_length)
        env.update_task()
        print(f"New task: {env.current_task.copy()}")
    env.close()
    plt.ioff()
    plt.close()


def test_update_task():
    """Test that using update_task changes the given values in the environment
    and in the current_task dict, and that when a value isn't passed to
    update_task, it isn't reset to its default but instead keeps its previous
    value. 
    """
    original = gym.make("CartPole-v0")
    env = MultiTaskEnvironment(original)
    env.reset()
    env.seed(123)

    assert env.length == original.length
    env.update_task(length=1.0)
    assert env.current_task["length"] == env.length == 1.0
    env.update_task(gravity=20.0)
    assert env.length == 1.0
    assert env.current_task["gravity"] == env.gravity == 20.0
    env.close()