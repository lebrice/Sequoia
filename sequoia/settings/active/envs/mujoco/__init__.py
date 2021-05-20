from .modified_gravity import ModifiedGravityEnv
from .modified_size import ModifiedSizeEnv
from .half_cheetah import HalfCheetahEnv, ContinualHalfCheetahEnv
from .hopper import HopperEnv#, ContinualHopperEnv
import gym
from gym.envs import register
# id (str): The official environment ID
# entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
# reward_threshold (Optional[int]): The reward threshold before the task is considered solved
# nondeterministic (bool): Whether this environment is non-deterministic even after seeding
# max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of
# kwargs (dict): The kwargs to pass to the environment class
import os
from pathlib import Path

SOURCE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

from typing import Type

def get_entry_point(Env: Type[gym.Env]) -> str:
    return f"{Env.__module__}:{Env.__name__}"

gym.envs.register(
    id="ContinualHalfCheetah-v0",
    entry_point=get_entry_point(ContinualHalfCheetahEnv),
    max_episode_steps=1000,
    reward_threshold=3800.0,
    # TODO: Not using this, but we could if we wanted to register one env per task.
    # kwargs=dict(body_parts=['torso','fthigh','fshin','ffoot'], size_scales=size_factors)
)
