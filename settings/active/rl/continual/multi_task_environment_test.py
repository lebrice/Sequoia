from typing import Dict, List

import gym
import matplotlib.pyplot as plt
import pytest

from .multi_task_environment import MultiTaskEnvironment


@pytest.mark.parametrize("environment_name", ["CartPole-v0"])
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
        env.reset(new_task=True)
        print(f"New task: {env.current_task_dict}")
    env.close()
    plt.ioff()
    plt.close()

from gym.envs.classic_control import CartPoleEnv

@pytest.mark.parametrize("environment_name", ["CartPole-v0"])

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

            task_param_values.append(env.current_task_dict())
            # env.update_task(length=(i + 1) / 100 * 2 * default_length)
        env.update_task()
        print(f"New task: {env.current_task_dict()}")
    env.close()
    plt.ioff()
    plt.close()

