import inspect
import math
import os
import os.path as osp
import random
import tempfile
import xml.etree.ElementTree as ET
from typing import ClassVar, Dict, List, Sequence, Tuple, Union

import gym
import mujoco_py
import numpy as np
import six
from gym import utils
from gym.envs.mujoco import MujocoEnv


class ModifiedSizeEnv(MujocoEnv):
    """
    Allows changing the size of the body parts.

    TODO: This currently can modify the geometry in-place (at least visually) with the
    `self.model.geom_size` ndarray, but the joints don't follow the change in length.
    """
    BODY_NAMES: ClassVar[List[str]]

    # IDEA: Use somethign like this to tell appart modifications which can be applied
    # on-the-fly on a given env to get multiple tasks, vs those that require creating a
    # new environment for each task.
    CAN_BE_UPDATED_IN_PLACE: ClassVar[bool] = False

    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        # TODO: IF using one or more of these `Modified<XYZ>` buffers, then we need to
        # get each one a distinct argument name, which isn't ideal!
        body_parts: List[str] = None,  # Has to be the name of a geom, not of a body!
        size_scales: List[float] = None,
        body_name_to_size_scale: Dict[str, float] = None,
        **kwargs,
    ):
        body_parts = body_parts or []
        size_scales = size_scales or []
        body_name_to_size_scale = body_name_to_size_scale or {}
        body_name_to_size_scale.update(zip(body_parts, size_scales))

        # super().__init__(model_path=model_path, frame_skip=frame_skip)

        if model_path.startswith("/"):
            full_path = model_path
        else:
            full_path = os.path.join(
                os.path.dirname(inspect.getsourcefile(MujocoEnv)), "assets", model_path
            )
        if not os.path.exists(full_path):
            raise IOError(f"File {full_path} does not exist")

        # find the body_part we want

        tree = ET.parse(full_path)
        if any(scale_factor == 0 for scale_factor in size_scales):
            raise RuntimeError(f"Can't use a scale_factor of 0!")

        if body_name_to_size_scale:
            print(f"Default XML path: {full_path}")
            # NOTE: For now this still modifies `tree` in-place.
            tree = change_size_in_xml(input_tree=tree, **body_name_to_size_scale)
            # create new xml
            _, new_xml_path = tempfile.mkstemp(suffix=".xml", text=True)
            tree.write(new_xml_path)
            print(f"Generated XML path: {new_xml_path}")
            full_path = new_xml_path

        # idx = self.model.body_names.index(six.b(body_name))
        # temp = np.copy(self.model.geom_size)

        # load the modified xml
        super().__init__(model_path=full_path, frame_skip=frame_skip, **kwargs)
        if body_name_to_size_scale:
            print(f"Modifying size of body parts: {body_name_to_size_scale}")
            resulting_sizes = {
                    k: v
                    for k, v in self.get_size_dict().items()
                    if k in body_name_to_size_scale
                }
            print(f"Resulting sizes: {resulting_sizes}")

        # assert False, self.model.geom_size

    #     self.default_sizes_dict: Dict[str, np.ndarray] = {
    #         body_name: self.model.geom_size[i]
    #         for i, body_name in enumerate(self.model.body_names)
    #     }
    # self.default_sizes: np.ndarray = np.copy(self.model.geom_size)
    # self.default_geom_rbound: np.ndarray = np.copy(self.model.geom_rbound)
    # self.scale_size(**dict(zip(body_parts, size_scales)))

    #     self._update()
    #     # self.model._compute_subtree()
    #     # self.model.forward()

    # def get_and_modify_bodysize(self, body_name: str, scale: float):
    #     idx = self.model.body_names.index(six.b(body_name))
    #     temp = np.copy(self.model.geom_size)
    #     temp[idx] *= scale
    #     self.sim = mujoco_py.MjSim(self.model)
    #     self.data = self.sim.data
    #     return temp

    # def _update(self) -> None:
    #     """'Update' the model, if necessary, after a change has occured to the model.

    #     TODO: Not sure if this is entirely correct
    #     """
    #     # TODO: FIgure this out.
    #     # self.model._compute_subtree()
    #     # self.model.forward()

    def get_size(self, body_part: str) -> np.ndarray:
        # Will raise an IndexError if the body part isnt found.
        if body_part not in self.model.geom_names:
            if f"{body_part}_geom" in self.model.geom_names:
                body_part = f"{body_part}_geom"
        assert body_part in self.model.geom_names, body_part
        idx = self.model.geom_names.index(body_part)
        size = self.model.geom_size[idx].copy()
        return size

    def get_size_dict(self) -> Dict[str, np.ndarray]:
        # TODO: There might be more than one <geom> per <body> element, we just return
        # the one with same name of with name + _geom
        return {
            body_name: self.get_size(body_name) for body_name in self.model.body_names
            if body_name in self.BODY_NAMES
        }

    # def _scale_size(self, body_part: str, factor: Union[float, np.ndarray]) -> None:
    #     # Will raise an IndexError if the body part isnt found.
    #     idx = self.model.body_names.index(body_part)
    #     geom_size = self.model.geom_size[idx]
    #     geom_rbound = self.model.geom_rbound[idx]
    #     self.model.geom_size[idx][:] = geom_size * factor
    #     self.model.geom_rbound[idx] = geom_rbound * factor
    #     # assert False,

    # def reset_sizes(self) -> None:
    #     """Resets the masses to their default values.
    #     """
    #     # NOTE: Use [:] to modify in-place, just in case there are any
    #     # pointer-shenanigans going on on the C side.
    #     self.model.geom_size[:] = self.default_sizes
    #     self.model.geom_rbound[:] = self.default_geom_rbound
    #     # self._update()

    # def scale_size(
    #     self,
    #     body_part: Union[str, Sequence[str]] = None,
    #     scale_factor: Union[float, Sequence[float]] = None,
    #     **body_name_to_scale_factor,
    # ) -> None:
    #     """ Set the size of the parts of the model, proportionally to their original
    #     size.
    #     """
    #     body_part = body_part or []
    #     scale_factor = scale_factor or []

    #     self.reset_sizes()
    #     body_name_to_scale_factor = body_name_to_scale_factor or {}

    #     body_parts = [body_part] if isinstance(body_part, str) else body_part
    #     scale_factors = (
    #         [scale_factor] if isinstance(scale_factor, float) else scale_factor
    #     )
    #     body_name_to_scale_factor.update(zip(body_parts, scale_factors))

    # for body_name, scale_factor in body_name_to_scale_factor.items():
    #     self._scale_size(body_name, scale_factor)

    # # Not sure if we need to do this?
    # self._update()

    # def get_and_modify_bodymass(self, body_name: str, scale: float):
    #     idx = self.model.body_names.index(six.b(body_name))
    #     temp = np.copy(self.model.body_mass)
    #     temp[idx] *= scale
    #     return temp


import copy


def pos_to_str(pos: Tuple[float, ...]) -> str:
    return " ".join("0" if v == 0 else str(round(v, 5)) for v in pos)


def str_to_pos(pos_str: str) -> Tuple[float, ...]:
    return tuple([float(v) for v in pos_str.split()])


def scale_pos(
    pos: Tuple[float, ...], coefficients: Tuple[float, ...],
) -> Tuple[float, ...]:
    return tuple(
        [v * axis_scaling_coef for v, axis_scaling_coef in zip(pos, coefficients)]
    )


def change_size_in_xml(
    input_tree: ET.ElementTree,
    **body_part_to_scale_factor: Dict[
        str, Union[float, Tuple[float, float, float], np.ndarray]
    ],
) -> ET.ElementTree:
    tree = copy.deepcopy(input_tree)

    for body_part, size_scale in body_part_to_scale_factor.items():
        # torso = tree.find(".//body[@name='%s']" % body_part)
        axis_scaling_coefs: np.ndarray
        if isinstance(size_scale, (int, float)):
            assert size_scale != 0, "HUH?"
            axis_scaling_coefs = np.asfarray([size_scale, size_scale, size_scale])
        else:
            assert len(size_scale) == 3
            axis_scaling_coefs = (
                np.asfarray(size_scale)
                if not isinstance(size_scale, np.ndarray)
                else size_scale
            )

        parent_body = tree.find(f".//body[@name='{body_part}']")
        # if parent_body is None:
        #     parent_body = tree.find(f".//body[@name='{body_part}_goem']")
        assert parent_body is not None, f"Can't find a body element that matches the name {body_part}"

        # grab the target geometry
        # target_geom = parent_body.find(f".//geom[@name='{body_part}']")
        child_bodies: List[ET.ElementTree] = parent_body.findall("./body") + parent_body.findall("./joint")
        # assert False, [(child_body.tag, child_body.attrib.get("name")) for child_body in child_bodies]
        # All child geometries that have a 'pos' attribute.
        child_geoms: List[ET.ElementTree] = parent_body.findall("./geom")
        # assert False, [(child_geom.tag, child_geom.attrib.get("name")) for child_geom in child_geoms]

        """TODO: Clean this up, use this as a template.

        <worldbody>
            <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
            <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane"/>
            <body name="torso" pos="0 0 1.25">
                <camera name="track" mode="trackcom" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
                <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
                <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
                <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 1.25" stiffness="0" type="hinge"/>
                <geom friction="0.9" fromto="0 0 1.45 0 0 1.05" name="torso_geom" size="0.05" type="capsule"/>
                <body name="thigh" pos="0 0 1.05">
                    <joint axis="0 -1 0" name="thigh_joint" pos="0 0 1.05" range="-150 0" type="hinge"/>
                    <geom friction="0.9" fromto="0 0 1.05 0 0 0.6" name="thigh_geom" size="0.05" type="capsule"/>
                    <body name="leg" pos="0 0 0.35">
                    <joint axis="0 -1 0" name="leg_joint" pos="0 0 0.6" range="-150 0" type="hinge"/>
                    <geom friction="0.9" fromto="0 0 0.6 0 0 0.1" name="leg_geom" size="0.04" type="capsule"/>
                    <body name="foot" pos="0.13/2 0 0.1">
                        <joint axis="0 -1 0" name="foot_joint" pos="0 0 0.1" range="-45 45" type="hinge"/>
                        <geom friction="2.0" fromto="-0.13 0 0.1 0.26 0 0.1" name="foot_geom" size="0.06" type="capsule"/>
                    </body>
                    </body>
                </body>
            </body>
        </worldbody>

        STEPS:
        1. Find the 'target' body (with the given name)
        2. Find all the geometries that are immediate children of that node.
        3. For each geometry:
            - if it has a 'fromto' attribute, scale the two endpoints:

                start -> new_start
                end -> new_end

            - if it has a 'pos' attribute, and ISNT the 'main' geom for the 'target'
              body, then scale that pos attribute:

                pos -> new_pos

            - if it has a `size` attribute, scale it:

                size -> new_size

        4. For each <body> that is a child of the target body:

            - If its `pos` matches the value of `start`, then set its `pos -> new_start`
            - If its `pos` matches the value of `end`, then set its `pos -> new_end`
            - If its `pos` matches the value of `pos`, then set its `pos -> new_pos`

            For all its geometries:
                - If it has a `pos` attribute that matches the value of `start`, then set its `pos -> new_start`
                - If it has a `pos` attribute that matches the value of `end`, then set its `pos -> new_end`
                - If it has a `pos` attribute that matches the value of `pos`, then set its `pos -> new_pos`
                - If it has a `fromto` attribute:
                    child_start, child_end = fromto[:3], fromto[3:]

                    - if `child_start` matches the value of `start`, then set `child_start = new_start`
                    - if `child_start` matches the value of `end`, then set `child_start = new_end`
                    - if `child_start` matches the value of `pos`, then set `child_start = new_pos`

                    - if `child_end` matches the value of `start`, then set `child_end = new_start`
                    - if `child_end` matches the value of `end`, then set `child_end = new_end`
                    - if `child_end` matches the value of `pos`, then set `child_end = new_pos`

        """
        target_body = tree.find(".//body[@name='{body_part}'])
        assert False, target_body



        for child_geom in child_geoms:
            if "fromto" in child_geom.attrib:
                # TODO: Only do this for the target geom, no?
                from_to = [float(v) for v in child_geom.attrib["fromto"].split()]
                start_point: Tuple[float, float, float] = tuple(from_to[:3])
                end_point: Tuple[float, float, float] = tuple(from_to[3:])
                # Useful for changing the child nodes below.

                new_start_point = scale_pos(start_point, axis_scaling_coefs)
                new_end_point = scale_pos(end_point, axis_scaling_coefs)

                new_start_point_str = pos_to_str(new_start_point)
                new_end_point_str = pos_to_str(new_end_point)

                new_from_to = new_start_point_str + " " + new_end_point_str
                print(
                    f"Changing the 'fromto' of {child_geom.tag} element {child_geom.attrib} to {new_from_to}"
                )
                child_geom.attrib["fromto"] = new_from_to

                def update_pos(element: ET.ElementTree):
                    pos = str_to_pos(element.attrib["pos"])
                    # TODO: The head in half-cheetah is still misplaced when the torso
                    # is made bigger, but oh well!
                    # assert False, (child_body.attrib.get("name"), pos, start_point, pos == start_point)
                    if pos == start_point:
                        print(
                            f"Changing the 'pos' of {element.tag} element {element.attrib} to {new_start_point_str}"
                        )
                        element.attrib["pos"] = new_start_point_str
                    elif pos == end_point:
                        print(
                            f"Changing the 'pos' of {element.tag} element {element.attrib} to {new_end_point_str}"
                        )
                        element.attrib["pos"] = new_end_point_str



                for child_body in child_bodies:


                    if "pos" in child_body.attrib:
                        # TODO: This is ugly, but we want to also change the joint of
                        # the child body, if it's connected to the anchor point.
                        update_pos(child_body)
                        for child_child_element in child_body.getchildren():
                            if "pos" in child_child_element.attrib:
                                assert False, child_child_element.attrib
                                update_pos(child_child_element)

                    # <body name="thigh" pos="0 0 2.1">
                    #     <joint axis="0 -1 0" name="thigh_joint" pos="0 0 1.05" range="-150 0" type="hinge" />
                    #     <geom friction="0.9" fromto="0 0 1.05 0 0 0.6" name="thigh_geom" size="0.05" type="capsule" />
                    #     <body name="leg" pos="0 0 0.35">


            if "size" in child_geom.attrib:
                # Scale the 'size' attribute?
                sizes = [float(x) for x in child_geom.attrib["size"].split()]
                # rescale
                if len(sizes) != len(axis_scaling_coefs):
                    # Here we're trying to scale the 'size' attribute, but we have
                    if len(set(axis_scaling_coefs)) != 1:
                        raise RuntimeError(
                            f"Don't know how to scale size {sizes} given coefficients "
                            f"{axis_scaling_coefs}"
                        )
                new_sizes = [
                    v * axis_scaling_coef
                    for v, axis_scaling_coef in zip(sizes, axis_scaling_coefs)
                ]
                new_sizes_str = " ".join(str(v) for v in new_sizes)
                print(
                    f"Changing the 'size' of {child_geom.tag} element {child_geom.attrib} to {new_sizes}"
                )
                child_geom.attrib["size"] = new_sizes_str

            if "pos" in child_geom.attrib:
                # TODO: sort-of a hack: We don't want to modify the pos of the 'target'
                # geom, but we want to scale the pos of the other geoms. For instance,
                # we want to move the 'head' in half_cheetah.
                if child_geom.attrib["name"] in {body_part, f"{body_part}_goem"}:
                    continue
                pos = str_to_pos(child_geom.attrib["pos"])
                new_pos = scale_pos(pos, axis_scaling_coefs)
                print(
                    f"Changing the 'pos' of {child_geom.tag} element {child_geom.attrib} to {new_pos}"
                )
                child_geom.attrib["pos"] = pos_to_str(new_pos)
                # scale the 'pos' of the child geom.

        # for child_body in child_bodies:
        #     if "pos" in child_geom.attrib and child_geom.attrib["name"] != body_part:
        #         pos = str_to_pos(child_geom.attrib["pos"])
        #         new_pos = scale_pos(pos, axis_scaling_coefs)
        #         child_geom.attrib["pos"] = pos_to_str(new_pos)
        # TODO: in the future we want to also be able to make it longer or shorter,
        # but this requires propagation of the fromto attribute so like a middle
        # part isn't super long but the other parts connect at the same spot.
        # -.5 0 0 .5 0 0

        # TODO: Find all 'child' elements that are also bodies, and change their 'pos' attribute.
        # geom..".//geom[@name='{body_part}']"

        # fromto = []
        # for x in geoms[0].attrib["fromto"].split(" "):
        #     fromto.append(float(x))
        # fromto = [x*length_scale for x in fromto]
        # geoms[0].attrib["fromto"] = str() * length_scale) # the first one should always be the thing we want.
    return tree

