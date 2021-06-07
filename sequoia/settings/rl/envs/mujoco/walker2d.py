from typing import ClassVar, List, Dict, Tuple

from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.walker2d import Walker2dEnv as _Walker2dV2Env
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv as _Walker2dV3Env
from .modified_gravity import ModifiedGravityEnv
from .modified_mass import ModifiedMassEnv
from .modified_size import ModifiedSizeEnv


class Walker2dV2Env(_Walker2dV2Env):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the
    xml name as a kwarg in openai gym
    """

    BODY_NAMES: ClassVar[List[str]] = [
        "torso",
        "thigh",
        "leg",
        "foot",
        "thigh_left",
        "leg_left",
        "foot_left",
    ]

    def __init__(self, model_path: str = "walker2d.xml", frame_skip: int = 4):
        MujocoEnv.__init__(self, model_path=model_path, frame_skip=frame_skip)


class Walker2dV3Env(_Walker2dV3Env):
    BODY_NAMES: ClassVar[List[str]] = [
        "torso",
        "thigh",
        "leg",
        "foot",
        "thigh_left",
        "leg_left",
        "foot_left",
    ]

    def __init__(
        self,
        model_path: str = "walker2d.xml",
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.8, 2.0),
        healthy_angle_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        xml_file: str = None,
        frame_skip: int = 4,
    ):
        if frame_skip != 4:
            raise NotImplementedError("todo: Add a frame_skip arg to the gym class.")
        super().__init__(
            xml_file=xml_file or model_path,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            healthy_angle_range=healthy_angle_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        )


class Walker2dGravityEnv(ModifiedGravityEnv, Walker2dV2Env):
    # NOTE: This environment could be used in ContinualRL!
    def __init__(
        self,
        model_path: str = "walker2d.xml",
        frame_skip: int = 4,
        gravity: float = -9.81,
    ):
        super().__init__(model_path=model_path, frame_skip=frame_skip, gravity=gravity)


class ContinualWalker2dV2Env(
    ModifiedGravityEnv, ModifiedSizeEnv, ModifiedMassEnv, Walker2dV2Env
):
    def __init__(
        self,
        model_path: str = "walker2d.xml",
        frame_skip: int = 4,
        gravity=-9.81,
        body_parts=None,  # 'torso_geom','thigh_geom','leg_geom','foot_geom'
        size_scales=None,  # (1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            gravity=gravity,
            body_parts=body_parts,
            size_scales=size_scales,
        )


class ContinualWalker2dV3Env(
    ModifiedGravityEnv, ModifiedSizeEnv, ModifiedMassEnv, Walker2dV3Env
):
    # def __init__(self, model_path, frame_skip, gravity=-9.81, **kwargs):
    #     super().__init__(model_path, frame_skip, gravity=gravity, **kwargs)
    def __init__(
        self,
        model_path: str = "walker2d.xml",
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.8, 2.0),
        healthy_angle_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        gravity=-9.81,
        body_name_to_size_scale: Dict[str, float] = None,
        xml_file: str = None,
        frame_skip: int = 4,
    ):
        if frame_skip != 4:
            raise NotImplementedError("todo: Add a frame_skip arg to the gym class.")
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            xml_file=xml_file or model_path,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            healthy_angle_range=healthy_angle_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            body_name_to_size_scale=body_name_to_size_scale,
            gravity=gravity,
        )
