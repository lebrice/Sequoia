# TODO: Should we use HopperV3 instead?
from gym.envs.mujoco.hopper import HopperEnv as _HopperEnv
from gym.envs.mujoco import MujocoEnv
from typing import ClassVar, List, Dict
from .modified_gravity import ModifiedGravityEnv
from .modified_size import ModifiedSizeEnv
from .modified_mass import ModifiedMassEnv

# NOTE: Removed the `utils.EzPickle` base class (since it wasn't being passed any kwargs
# (and therefore wasn't saving any of the 'state') anyway.


class HopperEnv(_HopperEnv):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the
    xml name as a kwarg in openai gym
    """
    BODY_NAMES: ClassVar[List[str]] = ["torso", "thigh", "leg", "foot"]

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


class ContinualHopperEnv(
    ModifiedGravityEnv, ModifiedSizeEnv, ModifiedMassEnv, HopperEnv
):
    def __init__(
        self,
        model_path: str = "hopper.xml",
        frame_skip: int = 4,
        gravity=-9.81,
        body_parts=None,  # 'torso_geom','thigh_geom','leg_geom','foot_geom'
        size_scales=None,  # (1.0, 1.0, 1.0, 1.0),
        body_name_to_size_scale: Dict[str, float] = None,
    ):
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            gravity=gravity,
            body_parts=body_parts,
            size_scales=size_scales,
            body_name_to_size_scale=body_name_to_size_scale,
        )
