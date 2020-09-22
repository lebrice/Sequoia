
import multiprocessing as mp
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection, wait
from typing import Any, List, Union

import gym
from baselines.common.vec_env import CloudpickleWrapper, VecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from utils.logging_utils import get_logger

logger = get_logger(__file__)
process_name = mp.current_process().name
print(f"Current process name: {process_name}")

class Commands:
    step = "step"
    reset = "reset"
    render = "render"
    close = "close"
    get_spaces_spec = "get_spaces_spec"
    get_attr = "getattr"
    set_attr = "setattr"
    set_attr_on_each = "setattr_on_each"
    partial_reset = "partial_reset"
    seed = "seed"


def custom_worker(remote: Connection,
                  parent_remote: Connection,
                  env_fn_wrappers: CloudpickleWrapper,
                  worker_index: int = None,
                  auto_reset: bool = True):
    """Copied this from the baselines package, slightly modifying it to accept
    other commands.
    """
    idx = "" if worker_index is None else worker_index

    def step_env(env, action):
        observation, reward, done, info = env.step(action)
        # TODO: Is this really what we want to do?
        if done and auto_reset:
            observation = env.reset()
        return observation, reward, done, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    
    timeout: int = 1
    timeout_count: int = 0
    # After how many timeouts to close the connection.
    death: int = 5

    try:
        while True:
            # TODO: This seemed necessary before, but I'm not sure anymore.
            ready_objects: List = wait([remote], timeout=timeout)
            if not ready_objects:
                logger.debug(f"Worker {idx} hasn't received a command over the last "
                             f"{timeout * timeout_count} seconds.")
                timeout_count += 1
                if timeout_count == death:
                    remote.close()
                    break
                continue
            cmd, *data = remote.recv()
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], CloudpickleWrapper):
                data = data[0].x
            logger.debug(f"Worker {idx} received command {cmd}, data {data}")
            
            if cmd == Commands.step:
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            
            elif cmd == Commands.reset:
                remote.send([env.reset() for env in envs])
            
            elif cmd == Commands.render:
                remote.send([env.render(mode='rgb_array') for env in envs])
            
            elif cmd == Commands.close:
                remote.close()
                break
            
            elif cmd == Commands.get_spaces_spec:
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space, envs[0].spec)))
            
            elif cmd == Commands.get_attr:
                # Adding this: When asked to get an attribute, get the attr.
                sentinel = object()
                attributes: List[Union[Any, AttributeError]] = []
                # Attribute name, and sentinel value used for the default.
                name: str = data
                default = sentinel
                if isinstance(data, str):
                    name = data
                elif len(data) == 1:
                    # single-item tuple with just the attr name.
                    name = data[0]
                elif len(data) == 2:
                    # default value was supplied.
                    name = data[0]
                    default = data[1]

                for env in envs:
                    # Actually get the attribute.
                    # NOTE: When the attribute isn't found on the env, we
                    # return the exception. I'm trying this out so that we
                    # don't crash the worker when an AttributeError occurs.
                    # TODO: Could also use a sentinel default value, so we never
                    # even cause AttributeErrors in the envs. However, then what
                    # do we return if we don't find the attribute?
                    try:
                        attribute = getattr(env, name, default)
                    except AttributeError as exc:
                        attribute = exc
                    else:
                        if attribute is sentinel:
                            attribute = AttributeError(f"Env {env} doesn't have a '{name}' attribute.")
                    attributes.append(attribute)
                remote.send(CloudpickleWrapper(attributes))

            elif cmd == Commands.set_attr:
                # Adding this: When asked to set an attribute, set the attr.
                # NOTE: this sets the same value for all envs.
                if isinstance(data, list) and len(data) == 1:
                    data = data[0]
                if isinstance(data, CloudpickleWrapper):
                    data = data.x
                assert len(data) == 2
                attr_name = data[0]
                attr_value = data[1]
                for env in envs:
                    wrapper = set_attr_on_env(env, attr_name, attr_value)
                    logger.debug(f"(Set the attribute on the {wrapper} wrapper.")
                remote.send([None for env in envs])
            
            elif cmd == Commands.set_attr_on_each:
                # IDEA: Instead of setting the same value for all envs, here
                # `data` must have length `len(envs)`, so we can just split it
                # and set a different value for each env.
                if isinstance(data, (list, tuple)) and len(data) == 1:
                    data = data[0]
                if isinstance(data, CloudpickleWrapper):
                    data = data.x
                assert len(data) == 2
                attr_name = data[0]
                attr_values = data[1]
                assert len(attr_values) == len(envs)
                for i, (env, attr_value) in enumerate(zip(envs, attr_values)):
                    logger.debug(f"Worker {idx}: envs[{i}].{attr_name} = {attr_value}")
                    wrapper = set_attr_on_env(env, attr_name, attr_value)
                    logger.debug(f"(Set the attribute on the {wrapper} wrapper.")
                remote.send([None for env in envs])
            
            elif cmd == Commands.partial_reset:
                # TODO: Idea, if data[i] is True, reset envs[i].
                assert len(data) == len(envs)
                states = []
                for env, reset in zip(envs, data):
                    if reset:
                        state = env.reset()
                    else:
                        state = None
                    states.append(state)
                remote.send(CloudpickleWrapper(states))

            elif cmd == Commands.seed:
                # TODO: Idea, if data[i] is True, reset envs[i].
                assert len(data) == len(envs)
                for env, seed in zip(envs, data):
                    env.seed(seed)
                remote.send([None for env in envs])

            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
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
