"""Customized version of the worker_ function from gym.vector.async_vector_env.
"""
import multiprocessing as mp
import sys
from multiprocessing.connection import Connection, wait
from typing import Any, List, Union

import gym
from gym.vector.async_vector_env import write_to_shared_memory
from gym.vector.async_vector_env import _worker, _worker_shared_memory
from gym.vector.utils import CloudpickleWrapper

# TODO: Find a way to turn off the logs coming from the workers. 
# from utils.logging_utils import get_logger


class Commands:
    step = "step"
    reset = "reset"
    render = "render"
    close = "close"

    # WIP:
    apply = "apply"

    # Things to re-add:
    get_attr = "getattr"
    set_attr = "setattr"
    set_attr_on_each = "setattr_on_each"
    partial_reset = "partial_reset"
    seed = "seed"


from gym import Env
from typing import Callable
from multiprocessing.queues import Queue

def _custom_worker_shared_memory(index: int,
                                 env_fn: Callable[[], Env],
                                 pipe: Connection,
                                 parent_pipe: Connection,
                                 shared_memory,
                                 error_queue: Queue,
                                 in_series: int = None):
    """Copied and modified from `gym.vector.async_vector_env`.

    Args:
        index ([type]): [description]
        env_fn ([type]): [description]
        pipe ([type]): [description]
        parent_pipe ([type]): [description]
        shared_memory ([type]): [description]
        error_queue ([type]): [description]
        **new**: (TODO:)
        in_series: (int, optional): When passed, we create `in_series`
        environments in series in this worker.
    Raises:
        RuntimeError: [description]
    """
    process_name = mp.current_process().name
    # print(f"Current process name: {process_name}")

    assert shared_memory is not None
    # if in_series is not None:
    #     # TODO: Would need to un-flatten the chunks in the class that uses this.
    #     env = gym.vector.SyncVectorEnv([env_fn for i in range(in_series)])
    # else:
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            # print(f"Worker {index} received command {command}")
            if command == Commands.reset:
                observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send((None, True))
            elif command == Commands.step:
                observation, reward, done, info = env.step(data)
                if done:
                    if isinstance(info, dict):
                        info["final_state"] = observation
                    observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send(((None, reward, done, info), True))
            elif command == Commands.seed:
                env.seed(data)
                pipe.send((None, True))
            elif command == Commands.close:
                pipe.send((None, True))
                break
            elif command == '_check_observation_space':
                pipe.send((data == observation_space, True))

            # Below this: added commands.
            elif command == Commands.apply:
                assert callable(data)
                function = data
                results = function(env)
                pipe.send((results, True))

            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, '
                    '`_check_observation_space`, `apply`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _custom_worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                pipe.send((observation, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_check_observation_space':
                pipe.send((data == env.observation_space, True))
            
            # Below this: added commands.
            elif command == Commands.apply:
                assert callable(data)
                function = data
                results = function(env)
                pipe.send((results, True))
            
            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, '
                    '`_check_observation_space`, `apply`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send((None, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send(((None, reward, done, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_check_observation_space':
                pipe.send((data == observation_space, True))
            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, '
                    '`_check_observation_space`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()

def set_attr_on_env(env: Union[gym.Env, gym.Wrapper], attr: str, value: Any) -> Union[gym.Env, gym.Wrapper]:
    """ Sets the attribute `attr` to a value of `value` on the first wrapper
    that already has it.
    If none have it, sets the attribute on the unwrapped env.

    Returns the env or wrapper on which the attribute was set.
    """
    while hasattr(env, "env") and not hasattr(env, attr):
        env = env.env
    setattr(env, attr, value)
    return env
