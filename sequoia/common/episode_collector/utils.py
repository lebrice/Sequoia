from sequoia.common.typed_gym import _Env, _Observation, _Action, _Reward, _Space
import numpy as np
import gym
from gym import spaces
from gym.vector import VectorEnv
from typing import Tuple, Optional


def get_reward_space(env: _Env[_Observation, _Action, _Reward]) -> _Space[_Reward]:
    if hasattr(env, "reward_space") and env.reward_space is not None:
        return env.reward_space
    reward_range: Tuple[float, float] = getattr(env, "reward_range", (-np.inf, np.inf))
    num_envs = env.num_envs if isinstance(env.unwrapped, VectorEnv) else None
    return spaces.Box(
        reward_range[0],
        reward_range[1],
        dtype=float,
        shape=(num_envs,) if num_envs is not None else (),
    )


def get_num_envs(env: _Env) -> Optional[int]:
    if isinstance(env.unwrapped, VectorEnv):
        return env.num_envs
    else:
        return None
