import inspect
import math
import os
import os.path as osp
import random
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Union, Sequence, ClassVar

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

    # IDEA: Use somethign like this to tell appart modifications which can be applied
    # on-the-fly on a given env to get multiple tasks, vs those that require creating a
    # new environment for each task.
    CAN_BE_UPDATED_IN_PLACE: ClassVar[bool] = False

    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        body_parts: List[str] = None,
        size_scales: List[float] = None,
        **kwargs,
    ):
        body_parts = body_parts or []
        size_scales = size_scales or []
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
        
        scaling_factor_dict = dict(zip(body_parts, size_scales))

        # NOTE: For now this still modifies `tree` in-place.
        tree = change_size_in_xml(input_tree=tree, **scaling_factor_dict)
        # create new xml
        _, file_path = tempfile.mkstemp(suffix=".xml", text=True)
        tree.write(file_path)

        # idx = self.model.body_names.index(six.b(body_name))
        # temp = np.copy(self.model.geom_size)

        # load the modified xml
        super().__init__(model_path=file_path, frame_skip=frame_skip, **kwargs)
        if body_parts:
            print(f"Generated XML path: {file_path}")
            print(f"Modifying size of body parts: {scaling_factor_dict}")
            print(f"Resulting sizes: ", {k: v for k, v in self.get_size_dict().items() if k in body_parts})

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
        idx = self.model.body_names.index(body_part)
        size = self.model.geom_size[idx]
        return size

    def get_size_dict(self) -> Dict[str, np.ndarray]:
        return {
            body_name: self.model.geom_size[i].copy()
            for i, body_name in enumerate(self.model.body_names)
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
    pos: Tuple[float, ...],
    coefficients: Tuple[float, ...],
) -> Tuple[float, ...]:
    return tuple(
        [
            v * axis_scaling_coef
            for v, axis_scaling_coef in zip(pos, coefficients)
        ]
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
        if isinstance(size_scale, float):
            assert size_scale != 0, f"HUH?!"
            axis_scaling_coefs = np.asfarray([size_scale, size_scale, size_scale])
        else:
            assert len(size_scale) == 3
            axis_scaling_coefs = (
                np.asfarray(size_scale)
                if not isinstance(size_scale, np.ndarray)
                else size_scale
            )
        parent_body = tree.find(f".//body[@name='{body_part}']")
        # grab the target geometry
        # target_geom = parent_body.find(f".//geom[@name='{body_part}']")
        child_bodies = parent_body.findall("./body")
        # assert False, [(child_body.tag, child_body.attrib.get("name")) for child_body in child_bodies]
        # All child geometries that have a 'pos' attribute.
        child_geoms: List[ET.ElementTree] = parent_body.findall("./geom")
        # assert False, [(child_geom.tag, child_geom.attrib.get("name")) for child_geom in child_geoms]


        for child_geom in child_geoms:
            if "fromto" in child_geom.attrib:
                from_to = [float(v) for v in child_geom.attrib["fromto"].split()]
                start_point: Tuple[float, float, float] = tuple(from_to[:3])
                end_point: Tuple[float, float, float] = tuple(from_to[3:])
                # Useful for changing the child nodes below.

                new_start_point = scale_pos(start_point, axis_scaling_coefs)
                new_end_point = scale_pos(end_point, axis_scaling_coefs)

                new_start_point_str = pos_to_str(new_start_point)
                new_end_point_str = pos_to_str(new_end_point)

                new_from_to = new_start_point_str + " " + new_end_point_str
                print(f"Changing the 'fromto' of {child_geom.tag} element {child_geom.attrib} to {new_from_to}")
                child_geom.attrib["fromto"] = new_from_to

                for child_body in child_bodies:
                    if "pos" in child_body.attrib:
                        pos = str_to_pos(child_body.attrib["pos"])
                        # TODO: The head in half-cheetah is still misplaced when the torso
                        # is made bigger, but oh well!
                        # assert False, (child_body.attrib.get("name"), pos, start_point, pos == start_point)
                        if pos == start_point:
                            print(f"Changing the 'pos' of {child_body.tag} element {child_body.attrib} to {new_start_point_str}")
                            child_body.attrib["pos"] = new_start_point_str
                        elif pos == end_point:
                            print(f"Changing the 'pos' of {child_body.tag} element {child_body.attrib} to {new_end_point_str}")
                            child_body.attrib["pos"] = new_end_point_str

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
                print(f"Changing the 'size' of {child_geom.tag} element {child_geom.attrib} to {new_sizes}")
                child_geom.attrib["size"] = new_sizes_str

            if "pos" in child_geom.attrib and child_geom.attrib["name"] != body_part:
                pos = str_to_pos(child_geom.attrib["pos"])
                new_pos = scale_pos(pos, axis_scaling_coefs)
                print(f"Changing the 'pos' of {child_geom.tag} element {child_geom.attrib} to {new_pos}")
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

