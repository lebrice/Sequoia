# TODO: Should we use HopperV3 instead?
from gym.envs.mujoco.hopper import HopperEnv as _HopperEnv
from gym.envs.mujoco import MujocoEnv
from .modified_gravity import ModifiedGravityEnv
from .modified_size import ModifiedSizeEnv


# NOTE: Removed the `utils.EzPickle` base class (since it wasn't being passed any kwargs
# (and therefore wasn't saving any of the 'state') anyway.


class HopperEnv(_HopperEnv):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the
    xml name as a kwarg in openai gym
    """

    def __init__(self, model_path: str = "hopper.xml", frame_skip: int = 4):
        MujocoEnv.__init__(self, model_path=model_path, frame_skip=frame_skip)
        # utils.EzPickle.__init__(self)


class HopperGravityEnv(ModifiedGravityEnv, HopperEnv):
    # NOTE: This environment could be used in ContinualRL!
    def __init__(
        self,
        model_path: str = "hopper.xml",
        frame_skip: int = 4,
        gravity: float = -9.81,
    ):
        super().__init__(model_path=model_path, frame_skip=frame_skip, gravity=gravity)

