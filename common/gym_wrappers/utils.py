import gym
from typing import Type

def wrapper_is_present(env: gym.Wrapper, wrapper_type: Type[gym.Wrapper]) -> bool:
    """Returns wether the given `env` has a wrapper of type `wrapper_type`. 

    Args:
        env (gym.Wrapper): a gym.Wrapper or a gym environment.
        wrapper_type (Type[gym.Wrapper]): A type of Wrapper to check for.

    Returns:
        bool: Wether there is a wrapper of that type wrapping `env`. 
    """
    while hasattr(env, "env"):
        if isinstance(env, wrapper_type):
            return True
        env = env.env
    return False
