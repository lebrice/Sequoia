""" TODO: Wrapper that modifies the friction, if possible on-the-fly. """
from typing import ClassVar
from gym.envs.mujoco import MujocoEnv


class ModifiedFrictionEnv(MujocoEnv):
    """
    Allows the gravity to be changed.
    
    Adapted from https://github.com/Breakend/gym-extensions/blob/master/gym_extensions/continuous/mujoco/gravity_envs.py
    """

    # IDEA: Use somethign like this to tell appart modifications which can be applied
    # on-the-fly on a given env to get multiple tasks, vs those that require creating a
    # new environment for each task.
    CAN_BE_UPDATED_IN_PLACE: ClassVar[bool] = True
