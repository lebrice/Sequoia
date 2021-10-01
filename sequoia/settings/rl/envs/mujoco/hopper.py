# TODO: Should we use HopperV3 instead?
from dataclasses import dataclass
from gym.envs.mujoco.hopper import HopperEnv as _HopperV2Env
from gym.envs.mujoco import MujocoEnv
from typing import ClassVar, List, Dict, NamedTuple, Tuple
from .modified_gravity import ModifiedGravityEnv
from .modified_size import ModifiedSizeEnv
from .modified_mass import ModifiedMassEnv

# NOTE: Removed the `utils.EzPickle` base class (since it wasn't being passed any kwargs
# (and therefore wasn't saving any of the 'state') anyway.

# TODO: Use HalfCheetah-v3 instead, which allows explicitly to change the model file!
from gym.envs.mujoco.hopper_v3 import HopperEnv as _HopperV3Env


class HopperV2Env(_HopperV2Env):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the
    xml name as a kwarg in openai gym
    """

    BODY_NAMES: ClassVar[List[str]] = ["torso", "thigh", "leg", "foot"]

    def __init__(self, model_path: str = "hopper.xml", frame_skip: int = 4):
        MujocoEnv.__init__(self, model_path=model_path, frame_skip=frame_skip)
        # utils.EzPickle.__init__(self)


class HopperV3Env(_HopperV3Env):
    BODY_NAMES: ClassVar[List[str]] = ["torso", "thigh", "leg", "foot"]

    def __init__(
        self,
        model_path="hopper.xml",
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
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
            healthy_state_range=healthy_state_range,
            healthy_z_range=healthy_z_range,
            healthy_angle_range=healthy_angle_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        )


class HopperV2GravityEnv(ModifiedGravityEnv, HopperV2Env):
    # NOTE: This environment could be used in ContinualRL!
    def __init__(
        self,
        model_path: str = "hopper.xml",
        frame_skip: int = 4,
        gravity: float = -9.81,
    ):
        super().__init__(model_path=model_path, frame_skip=frame_skip, gravity=gravity)


import inspect
import os
# from .modified_size import Pos, FromTo
from xml.etree.ElementTree import ElementTree, Element, parse
import copy
import tempfile


class ContinualHopperV2Env(ModifiedGravityEnv, ModifiedSizeEnv, ModifiedMassEnv, HopperV2Env):
    def __init__(
        self,
        model_path="hopper.xml",
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        xml_file: str = None,
        frame_skip: int = 4,
        gravity=-9.81,
        body_parts=None,  # 'torso_geom','thigh_geom','leg_geom','foot_geom'
        size_scales=None,  # (1.0, 1.0, 1.0, 1.0),
        body_name_to_size_scale: Dict[str, float] = None,
    ):
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            xml_file=xml_file or model_path,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_state_range=healthy_state_range,
            healthy_z_range=healthy_z_range,
            healthy_angle_range=healthy_angle_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            gravity=gravity,
            body_parts=body_parts,
            size_scales=size_scales,
            body_name_to_size_scale=body_name_to_size_scale,
        )


class ContinualHopperV3Env(ModifiedGravityEnv, ModifiedSizeEnv, ModifiedMassEnv, HopperV3Env):
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

# ------------- NOTE (@lebrice) -------------------------------
# Everything below this is unused.
# The idea was to do some kind of inverse-kinematics-ish math to fix the placement of the joints
# when the size of one of the parts of the model is changed.
#


# from typing import Dict


# def get_parent(tree: ElementTree, node: Element) -> Element:
#     parent_map: Dict[Element, Element] = {c: p for p in tree.iter() for c in p}
#     return parent_map[node]


# def update_world(
#     tree: ElementTree,
#     world_body: Element,
#     new_torso_max: Pos,
#     size_scaling_factor: float = 1.0,
#     **kwargs,
# ) -> None:
#     """propagate the changes from the body to the world, if need be."""
#     # TODO: Maybe move the camera etc?


# def update_torso(
#     tree: ElementTree = None,
#     torso_body: Element = None,
#     new_torso_min: Pos = None,
#     size_scaling_factor: float = 1.0,
#     geom_suffix="torso_geom",
#     **kwargs,
# ) -> None:
#     """'move' the torso body and its endpoints, after another bodypart has been
#     scaled.
#     This moves all relevant geoms and
#     joints and bodies,
#     Normally, this can update the
#     (through possibly recursive calls to one of `update_torso`,
#     `update_thigh`, `update_leg`, `update_foot`.)
#     """
#     assert size_scaling_factor != 0.0
#     body_name = "torso"
#     # Get the elements to be modified.
#     if torso_body is None:
#         assert tree is not None, "need the tree if torso_body is not given!"
#         if isinstance(tree, Element) and tree.tag == "body" and tree.get("name") == body_name:
#             torso_body = tree
#             tree = None
#         else:
#             torso_body = tree.find(f".//body[@name='{body_name}']")
#     assert torso_body is not None, "can't find the torso body!"

#     torso_geom = torso_body.find(f"./geom[@name='{body_name}']")
#     if torso_geom is None:
#         torso_geom = torso_body.find(f"./geom[@name='{body_name}_geom']")
#     if torso_geom is None:
#         raise RuntimeError(f"Can't find the geom for body part '{body_name}'!")

#     rooty_joint = torso_body.find("./joint[@name='rooty']")
#     rootz_joint = torso_body.find("./joint[@name='rootz']")

#     torso_body_pos = Pos.of_element(torso_body)

#     torso_geom_size = float(torso_geom.get("size"))
#     torso_geom_fromto = FromTo.of_element(torso_geom)
#     rootz_joint_ref = float(rootz_joint.get("ref"))
#     rooty_joint_pos = Pos.of_element(rooty_joint)

#     torso_max = torso_geom_fromto.start
#     torso_min = torso_geom_fromto.end
#     torso_length = torso_max - torso_min
#     assert torso_body_pos == torso_geom_fromto.center
#     # This happens to coincide with torso's pos.
#     assert rootz_joint_ref == torso_body_pos.z
#     assert rooty_joint_pos == torso_body_pos

#     if new_torso_min is None:
#         # Assume that the location of the base of the torso doesn't change, i.e. that
#         # this was called in order to JUST scale the torso and nothing else.
#         new_torso_min = torso_min
#     # new_torso_min is already given, calculate the other two:
#     new_torso_length = torso_length * (1 if size_scaling_factor is None else size_scaling_factor)
#     new_torso_max = new_torso_min + new_torso_length

#     # NOTE: fromto is from top to bottom here (maybe also everywhere else, not sure).
#     new_torso_geom_size = torso_geom_size * size_scaling_factor
#     new_torso_geom_fromto = FromTo(start=new_torso_max, end=new_torso_min)
#     new_torso_pos = (new_torso_max + new_torso_min) / 2
#     new_rootz_joint_ref = new_torso_pos.z
#     new_rooty_joint_pos = new_torso_pos

#     # Update the fields of the different elements.
#     torso_body.set("pos", new_torso_pos.to_str())
#     torso_geom.set("fromto", new_torso_geom_fromto.to_str())
#     torso_geom.set("size", new_torso_geom_size)

#     # TODO: Not sure if this makes sense: The rooty joint has a Pos that coincides
#     # with the torso pos.
#     new_torso_pos.set_in_element(rooty_joint)
#     # TODO: rootz has a 'ref' which also coincides with the torso pos.
#     rootz_joint.set("ref", str(new_rootz_joint_ref))
#     rooty_joint.set("pos", new_rooty_joint_pos)

#     new_torso_pos = new_torso_geom_fromto.center
#     # TODO: Also move the camera?

#     world_body: Optional[Element] = None
#     if tree is not None:
#         assert tree is not None, "need the tree if torso_body is not given!"
#         world_body = get_parent(tree, torso_body)

#     # Don't change the scaling of the parent, if this body part was scaled!
#     parent_scale_factor = 1 if size_scaling_factor != 1 else size_scaling_factor

#     update_world(
#         tree=tree,
#         world_body=world_body,
#         new_torso_min=new_torso_min,
#         new_torso_max=new_torso_max,
#         size_scaling_factor=parent_scale_factor,
#         **kwargs,
#     )


# def update_thigh(
#     tree: ElementTree = None,
#     thigh_body: Element = None,
#     new_thigh_min: Pos = None,
#     new_thigh_max: Pos = None,
#     size_scaling_factor: float = None,
#     **kwargs,
# ) -> None:
#     """'move' the thigh and its endpoints. This moves all relevant geoms and
#     joints and then moves the torso by calling `update_torso`.
#     """
#     # TODO:
#     new_torso_min = new_thigh_max
#     new_torso_max = todo

#     torso_body = get_parent(tree, thigh_body)
#     update_torso(
#         torso_body,
#         new_torso_min=new_torso_min,
#         new_torso_max=new_torso_max,
#         size_scaling_factor=size_scaling_factor,
#         new_thigh_min=new_thigh_min,
#         new_thigh_max=new_thigh_max,
#         **kwargs,
#     )


# def update_thigh(
#     tree: ElementTree = None,
#     thigh_body: Element = None,
#     new_thigh_min: Pos = None,
#     new_thigh_max: Pos = None,
#     size_scaling_factor: float = None,
#     **kwargs,
# ) -> None:
#     """'move' the thigh and its endpoints. This moves all relevant geoms and
#     joints and then moves the torso by calling `update_torso`.

#     """
#     new_torso_min = NotImplemented
#     new_thigh_max = NotImplemented
#     torso_body = get_parent(tree, thigh_body)
#     update_torso(
#         torso_body,
#         new_torso_min=new_torso_min,
#         size_scaling_factor=size_scaling_factor,
#         new_thigh_min=new_thigh_min,
#         new_thigh_max=new_thigh_max,  # Pass it in case the above components need it.
#         **kwargs,
#     )


# def scale_size(tree: ElementTree, body_name: str, scale: float) -> str:
#     tree = copy.deepcopy(tree)
#     target_body: Element = tree.find(f".//body[@name='{body_name}']")
#     parent_map: Dict[Element, Element] = {c: p for p in tree.iter() for c in p}

#     if body_name == "torso":
#         update_torso(tree, torso_body=target_body, size_scaling_factor=scale)
#     raise NotImplementedError(f"WIP")

