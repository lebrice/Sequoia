from typing import Dict, List

import gym
import matplotlib.pyplot as plt
import pytest
from gym import spaces
from gym.envs.classic_control import CartPoleEnv


from sequoia.common.spaces.named_tuple import NamedTuple, NamedTupleSpace
from sequoia.utils.utils import dict_union
from sequoia.conftest import monsterkong_required

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
        # env.render()
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
            # env.render()
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
            # env.render()
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
    

def test_add_task_dict_to_info():
    """ Test that the 'info' dict contains the task dict. """
    original: CartPoleEnv = gym.make("CartPole-v0")
    starting_length = original.length
    starting_gravity = original.gravity

    task_schedule = {
        10: dict(length=0.1),
        20: dict(length=0.2, gravity=-12.0),
        30: dict(gravity=0.9),
    }
    env = MultiTaskEnvironment(
        original,
        task_schedule=task_schedule,
        add_task_dict_to_info=True,
    )
    env.seed(123)
    env.reset()
    for step in range(100):
        _, _, done, info = env.step(env.action_space.sample())
        # env.render()
        if done:
            env.reset()

        if 0 <= step < 10:
            assert env.length == starting_length and env.gravity == starting_gravity
            assert info == env.default_task
        elif 10 <= step < 20:
            assert env.length == 0.1
            assert info == dict_union(env.default_task, task_schedule[10])
        elif 20 <= step < 30:
            assert env.length == 0.2 and env.gravity == -12.0
            assert info == dict_union(env.default_task, task_schedule[20])
        elif step >= 30:
            assert env.length == starting_length and env.gravity == 0.9
            assert info == dict_union(env.default_task, task_schedule[30])

    env.close()



def test_add_task_id_to_obs():
    """ Test that the 'info' dict contains the task dict. """
    original: CartPoleEnv = gym.make("CartPole-v0")
    starting_length = original.length
    starting_gravity = original.gravity

    task_schedule = {
        10: dict(length=0.1),
        20: dict(length=0.2, gravity=-12.0),
        30: dict(gravity=0.9),
    }
    env = MultiTaskEnvironment(
        original,
        task_schedule=task_schedule,
        add_task_id_to_obs=True,
    )
    env.seed(123)
    env.reset()
    
    assert env.observation_space == NamedTupleSpace(
        x=original.observation_space,
        task_labels=spaces.Discrete(4),   
    )
    
    
    for step in range(100):
        obs, _, done, info = env.step(env.action_space.sample())
        # env.render()

        x, task_id = obs
        
        if 0 <= step < 10:
            assert env.length == starting_length and env.gravity == starting_gravity
            assert task_id == 0, step

        elif 10 <= step < 20:
            assert env.length == 0.1
            assert task_id == 1, step
            
        elif 20 <= step < 30:
            assert env.length == 0.2 and env.gravity == -12.0
            assert task_id == 2, step
            
        elif step >= 30:
            assert env.length == starting_length and env.gravity == 0.9
            assert task_id == 3, step

        if done:
            obs = env.reset()
            assert isinstance(obs, tuple)


    env.close()


def test_starting_step_and_max_step():
    """ Test that when start_step and max_step arg given, the env stays within
    the [start_step, max_step] portion of the task schedule.
    """
    original: CartPoleEnv = gym.make("CartPole-v0")
    starting_length = original.length
    starting_gravity = original.gravity

    task_schedule = {
        10: dict(length=0.1),
        20: dict(length=0.2, gravity=-12.0),
        30: dict(gravity=0.9),
    }
    env = MultiTaskEnvironment(
        original,
        task_schedule=task_schedule,
        add_task_id_to_obs=True,
        starting_step=10,
        max_steps=19,
    )
    env.seed(123)
    env.reset()
    
    assert env.observation_space == NamedTupleSpace(
        x=original.observation_space,
        task_labels=spaces.Discrete(4),
    )

    # Trying to set the 'steps' to something smaller than the starting step
    # doesn't work.
    env.steps = -123
    assert env.steps == 10
    
    # Trying to set the 'steps' to something greater than the max_steps
    # doesn't work.
    env.steps = 50
    assert env.steps == 19

    # Here we reset the steps to 10, and also check that this works.
    env.steps = 10
    assert env.steps == 10
    
    for step in range(0, 100):
        # The environment started at an offset of 10.
        assert env.steps == max(min(step + 10, 19), 10)
        
        obs, _, done, info = env.step(env.action_space.sample())
        # env.render()

        x, task_id = obs

        # Check that we're always stuck between 10 and 20
        assert 10 <= env.steps < 20 
        assert env.length == 0.1
        assert task_id == 1, step

        if done:
            print(f"Resetting on step {step}")
            obs = env.reset()
            assert isinstance(obs, tuple)

    env.close()



def test_task_id_is_added_even_when_no_known_task_schedule():
    """ Test that even when the env is unknown or there are no task params, the
    task_id is still added correctly and is zero at all times.
    """
    # Breakout doesn't have default task params.
    original: CartPoleEnv = gym.make("Breakout-v0")
    env = MultiTaskEnvironment(
        original,
        add_task_id_to_obs=True,
    )
    env.seed(123)
    env.reset()
    
    assert env.observation_space == NamedTupleSpace(
        x=original.observation_space,
        task_labels=spaces.Discrete(1),
    )
    for step in range(0, 100):
        obs, _, done, info = env.step(env.action_space.sample())
        # env.render()

        x, task_id = obs
        assert task_id == 0

        if done:
            x, task_id = env.reset()
            assert task_id == 0
    env.close()



@monsterkong_required
def test_task_schedule_monsterkong():
    env: MetaMonsterKongEnv = gym.make("MetaMonsterKong-v1")
    from gym.wrappers import TimeLimit
    env = TimeLimit(env, max_episode_steps=10)
    env = MultiTaskEnvironment(env, task_schedule={
        0: {"level": 0},
        100: {"level": 1},
        200: {"level": 2},
        300: {"level": 3},
        400: {"level": 4},
    }, add_task_id_to_obs=True)
    obs = env.reset()
    
    # img, task_labels = obs
    assert obs[1] == 0
    assert env.get_level() == 0
    
    for i in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs[1] == i // 100
        assert env.level == i // 100
        env.render()
        assert isinstance(done, bool)
        if done:
            print(f"End of episode at step {i}")
            obs = env.reset()
    
    assert obs[1] == 4
    assert env.level == 4
    # level stays the same even after reaching that objective.
    for i in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs[1] == 4
        assert env.level == 4
        env.render()
        if done:
            print(f"End of episode at step {i}")
            obs = env.reset()

    env.close()


def test_task_schedule_with_callables():
    """ Apply functions to the env at a given step.
    
    """
    env: MetaMonsterKongEnv = gym.make("MetaMonsterKong-v1")
    from gym.wrappers import TimeLimit
    env = TimeLimit(env, max_episode_steps=10)
    
    from operator import methodcaller
    env = MultiTaskEnvironment(env, task_schedule={
        0:   methodcaller("set_level", 0),
        100: methodcaller("set_level", 1),
        200: methodcaller("set_level", 2),
        300: methodcaller("set_level", 3),
        400: methodcaller("set_level", 4),
    }, add_task_id_to_obs=True)
    obs = env.reset()
    
    # img, task_labels = obs
    assert obs[1] == 0
    assert env.get_level() == 0
    
    for i in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs[1] == i // 100
        assert env.level == i // 100
        env.render()
        assert isinstance(done, bool)
        if done:
            print(f"End of episode at step {i}")
            obs = env.reset()
    
    assert obs[1] == 4
    assert env.level == 4
    # level stays the same even after reaching that objective.
    for i in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs[1] == 4
        assert env.level == 4
        env.render()
        if done:
            print(f"End of episode at step {i}")
            obs = env.reset()