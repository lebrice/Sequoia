import ctypes
import warnings
from typing import ClassVar

from gym.envs.mujoco import MujocoEnv
from mujoco_py import MjSim


class ModifiedGravityEnv(MujocoEnv):
    """
    Allows the gravity to be changed.
    
    Adapted from https://github.com/Breakend/gym-extensions/blob/master/gym_extensions/continuous/mujoco/gravity_envs.py
    """

    # IDEA: Use somethign like this to tell appart modifications which can be applied
    # on-the-fly on a given env to get multiple tasks, vs those that require creating a
    # new environment for each task.
    CAN_BE_UPDATED_IN_PLACE: ClassVar[bool] = True

    def __init__(self, model_path: str, frame_skip: int, gravity: float = -9.81, **kwargs):
        super().__init__(model_path=model_path, frame_skip=frame_skip, **kwargs)
        # self.model.opt.gravity = (mujoco_py.mjtypes.c_double * 3)(*[0., 0., gravity])
        self.model.opt.gravity[:] = (ctypes.c_double * 3)(*[0.0, 0.0, gravity])
        # self.model._compute_subtree()
        # self.model.forward()
        self.sim.forward()
        self.sim: MjSim
        print(f"Setting initial gravity to {self.gravity}")
    
    @property
    def gravity(self) -> float:
        return self.model.opt.gravity[2]

    @gravity.setter
    def gravity(self, value: float) -> None:
        # TODO: Seems to be bad practice to modify memory in-place for some reason?
        self.model.opt.gravity[2] = value

    def set_gravity(self, value: float) -> None:
        if value >= 0:
            warnings.warn(RuntimeWarning(
                "Not a good idea to use a positive value! (things will start to float)"
            ))
        self.gravity = value
