from typing import ClassVar, List, Dict

import numpy as np
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as _HalfCheetahV2Env

from .modified_gravity import ModifiedGravityEnv
from .modified_mass import ModifiedMassEnv
from .modified_size import ModifiedSizeEnv


# TODO: Use HalfCheetah-v3 instead, which allows explicitly to change the model file!
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv as _HalfCheetahV3Env


class HalfCheetahV2Env(_HalfCheetahV2Env):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the
    xml name as a kwarg in openai gym
    """

    BODY_NAMES: ClassVar[List[str]] = [
        "torso",
        "bthigh",
        "bshin",
        "bfoot",
        "fthigh",
        "fshin",
        "ffoot",
    ]

    def __init__(self, model_path: str = "half_cheetah.xml", frame_skip: int = 5):
        MujocoEnv.__init__(self, model_path=model_path, frame_skip=frame_skip)


# Q: Why isn't HalfCheetahV3 based on HalfCheetahV2 in gym ?!


class HalfCheetahV3Env(_HalfCheetahV3Env):
    BODY_NAMES: ClassVar[List[str]] = [
        "torso",
        "bthigh",
        "bshin",
        "bfoot",
        "fthigh",
        "fshin",
        "ffoot",
    ]

    def __init__(
        self,
        model_path="half_cheetah.xml",
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        xml_file: str = None,
        frame_skip: int = 5,
    ):
        if frame_skip != 5:
            raise NotImplementedError("todo: Add a frame_skip arg to the gym class.")
        super().__init__(
            xml_file=xml_file or model_path,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        )


# class HalfCheetahGravityEnv(ModifiedGravityEnv, HalfCheetahEnv):
#     # NOTE: This environment could be used in ContinualRL!
#     def __init__(
#         self,
#         model_path: str = "half_cheetah.xml",
#         frame_skip: int = 5,
#         gravity: float = -9.81,
#     ):
#         super().__init__(model_path=model_path, frame_skip=frame_skip, gravity=gravity)


class HalfCheetahWithSensorEnv(HalfCheetahV2Env):
    """ NOTE: unused for now.
    Adds empty sensor readouts, this is to be used when transfering to WallEnvs where we
    get sensor readouts with distances to the wall
    """

    def __init__(self, model_path: str, frame_skip: int = 5, n_bins: int = 10):
        super().__init__(model_path=model_path, frame_skip=frame_skip)
        self.n_bins = n_bins

    def _get_obs(self):
        obs = np.concatenate(
            [
                super()._get_obs(),
                np.zeros(
                    self.n_bins
                ),  # NOTE: @lebrice HUH? what's the point of doing this?
                # goal_readings
            ]
        )
        return obs


# TODO: Rename these base classes to 'ModifyGravityMixin', 'ModifySizeMixin', etc.


class ContinualHalfCheetahV2Env(
    ModifiedGravityEnv, ModifiedSizeEnv, ModifiedMassEnv, HalfCheetahV2Env
):
    def __init__(
        self,
        model_path: str = "half_cheetah.xml",
        frame_skip: int = 5,
        gravity=-9.81,
        body_parts=None,  # ("torso", "fthigh", "fshin", "ffoot"),
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


class ContinualHalfCheetahV3Env(
    ModifiedGravityEnv, ModifiedSizeEnv, ModifiedMassEnv, HalfCheetahV3Env
):
    def __init__(
        self,
        model_path: str = "half_cheetah.xml",
        frame_skip: int = 5,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        gravity=-9.81,
        body_parts=None,  # ("torso", "fthigh", "fshin", "ffoot"),
        size_scales=None,  # (1.0, 1.0, 1.0, 1.0),
        body_name_to_size_scale: Dict[str, float] = None,
        xml_file: str = None,
    ):
        super().__init__(
            model_path=xml_file or model_path,
            frame_skip=frame_skip,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            gravity=gravity,
            body_parts=body_parts,
            size_scales=size_scales,
            body_name_to_size_scale=body_name_to_size_scale,
        )
