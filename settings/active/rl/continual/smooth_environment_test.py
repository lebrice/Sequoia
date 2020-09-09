import gym
import matplotlib.pyplot as plt
import pytest

from .smooth_environment import SmoothTransitions

@pytest.mark.parametrize("environment_name", ["CartPole-v0"])
def test_smooth_task_transitions(environment_name: str):   
    original = gym.make(environment_name)  
    env = SmoothTransitions(original)
    env.reset()
    env.seed(123)
    plt.ion()
    default_task = env.default_task
    starting_length: float = env.length
    for task_id in range(5):
        for i in range(10):
            observation, reward, done, info = env.step(env.action_space.sample())
            env.render()
            if done:
                env.reset(new_task=False)
        env.reset(new_task=True)

        print(f"New task: {dict(zip(env.task_params, env.current_task))}")
    env.close()
    plt.ioff()
    plt.close()
    assert False

from typing import List, Dict


@pytest.mark.parametrize("environment_name", ["CartPole-v0"])

def test_monitor_env(environment_name):
    original = gym.make(environment_name)
    # original = CartPoleEnv()
    env = SmoothTransitions(original)
    env = gym.wrappers.Monitor(env, f"recordings/smooth_{environment_name}", force=True)
    env.seed(123)
    env.reset()
    plt.ion()

    task_param_values: List[Dict] = []
    default_length: float = env.length
    from gym.wrappers import Monitor
    
    tasks = 10
    steps_per_task = 100
    total_steps = tasks * steps_per_task
    step = 0

    for task_id in range(tasks):
        for i in range(steps_per_task):
            step += 1
            # TODO: Here we change the length in real time, but we would instead
            # give a task schedule of some sort instead.
            env.update_task(length=(step / total_steps) * 2 * default_length)
            
            observation, reward, done, info = env.step(env.action_space.sample())
            env.render()

            if done:
                env.reset(new_task=False)

            task_param_values.append(env.current_task_dict())


        print(f"New task: {env.current_task_dict()}")
    env.close()
    plt.ioff()
    plt.close()
