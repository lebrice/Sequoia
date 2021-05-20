from gym.envs.mujoco.walker2d import Walker2dEnv as _Walker2dEnv
from gym.envs.mujoco import MujocoEnv
from .modified_gravity import ModifiedGravityEnv
from .modified_size import ModifiedSizeEnv


class Walker2dEnv(_Walker2dEnv):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the
    xml name as a kwarg in openai gym
    """
    def __init__(self, model_path: str = "walker2d.xml", frame_skip: int = 4):
        MujocoEnv.__init__(self, model_path=model_path, frame_skip=frame_skip)


class Walker2dGravityEnv(ModifiedGravityEnv, Walker2dEnv):
    # NOTE: This environment could be used in ContinualRL!
    def __init__(
        self,
        model_path: str = "walker2d.xml",
        frame_skip: int = 4,
        gravity: float = -9.81,
    ):
        super().__init__(model_path=model_path, frame_skip=frame_skip, gravity=gravity)
