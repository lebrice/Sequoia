from typing import ClassVar, List

import numpy as np
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as _HalfCheetahEnv

from .modified_gravity import ModifiedGravityEnv
from .modified_mass import ModifiedMassEnv
from .modified_size import ModifiedSizeEnv


class HalfCheetahEnv(_HalfCheetahEnv):
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


class HalfCheetahGravityEnv(ModifiedGravityEnv, HalfCheetahEnv):
    # NOTE: This environment could be used in ContinualRL!
    def __init__(
        self,
        model_path: str = "half_cheetah.xml",
        frame_skip: int = 5,
        gravity: float = -9.81,
    ):
        super().__init__(model_path=model_path, frame_skip=frame_skip, gravity=gravity)


class HalfCheetahWithSensorEnv(HalfCheetahEnv):
    """
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


class ContinualHalfCheetahEnv(
    ModifiedGravityEnv, ModifiedSizeEnv, ModifiedMassEnv, HalfCheetahEnv
):
    def __init__(
        self,
        model_path: str = "half_cheetah.xml",
        frame_skip: int = 5,
        gravity=-9.81,
        body_parts=None,  # ("torso", "fthigh", "fshin", "ffoot"),
        size_scales=None,  # (1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            gravity=gravity,
            body_parts=body_parts,
            size_scales=size_scales,
        )
