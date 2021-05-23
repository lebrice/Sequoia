import inspect
import math
import os
import os.path as osp
import random
import tempfile
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree
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
            raise NotImplementedError("TODO: Add this.")
            # tree = change_size_in_xml(input_tree=tree, **body_name_to_size_scale)
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
            body_name: self.get_size(body_name)
            for body_name in self.model.body_names
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

from typing import NamedTuple, Any
from collections import defaultdict


def pos_to_str(pos: Tuple[float, ...]) -> str:
    return " ".join("0" if v == 0 else str(round(v, 5)) for v in pos)


def str_to_pos(pos_str: str) -> "Pos":
    return Pos(*[float(v) for v in pos_str.split()])


class Pos(NamedTuple):
    x: float
    y: float
    z: float

    def to_str(self) -> str:
        """ Return the 'str' version of `self` to be placed in a 'pos' field in the XML.
        """
        return pos_to_str(self)

    @classmethod
    def from_str(cls, pos_str: str) -> "Pos":
        return cls(*[float(v) for v in pos_str.split()])

    def __mul__(self, value: Union[int, float, np.ndarray]) -> "Pos":
        if isinstance(value, (int, float)):
            value = [value for _ in range(len(self))]
        if not isinstance(value, (list, tuple, np.ndarray)):
            return NotImplemented
        assert len(value) == len(self)
        return type(self)(
            *[v * axis_scaling_coef for v, axis_scaling_coef in zip(self, value)]
        )

    def __eq__(self, other: Union[Tuple[float, ...], np.ndarray]):
        if not isinstance(other, (list, tuple, np.ndarray)):
            return NotImplemented
        return np.isclose(np.asfarray(self), np.asfarray(other)).all()

    def __rmul__(self, value: Any):
        return self * value

    def __truediv__(self, other: Union[int, float, Sequence[float]]):
        if isinstance(other, (int, float)):
            other = [other for _ in range(len(self))]
        if not isinstance(other, (list, tuple, np.ndarray)):
            return NotImplemented
        assert len(other) == len(self)
        return type(self)(*[v / v_other for v, v_other in zip(self, other)])

    def __add__(self, other: Union[int, float, np.ndarray]) -> "Pos":
        if isinstance(other, (int, float)):
            other = [other for _ in range(len(self))]
        if not isinstance(other, (list, tuple, np.ndarray)):
            return NotImplemented
        assert len(other) == len(self)
        return type(self)(*[v + v_other for v, v_other in zip(self, other)])

    def __radd__(self, other: Any) -> "Pos":
        return self + other

    def __neg__(self) -> "Pos":
        return type(self)(*[-v for v in self])

    def __sub__(self, other: Union[int, float, np.ndarray]) -> "Pos":
        if isinstance(other, (int, float)):
            other = [other for _ in range(len(self))]
        if not isinstance(other, (list, tuple, np.ndarray)):
            return NotImplemented
        assert len(other) == len(self)
        return self + (-other)
        # return type(self)(*[v + v_other for v, v_other in zip(self, other)])

    def __rsub__(self, other: Any) -> "Pos":
        return (-self) + other
    
    @classmethod
    def of_element(cls, element: Element, field: str = "pos") -> "Pos":
        if field not in element.attrib:
            raise RuntimeError(f"Element {element} doesn't have a '{field}' attribute.")
        return cls.from_str(element.attrib[field])

    def set_in_element(self, element: Element, field: str = "pos") -> None:
        if field not in element.attrib:
            # NOTE: Refusing to set a new field for now.
            raise RuntimeError(f"Element {element} doesn't have a '{field}' attribute.")
        element.set(field, self.to_str())

class FromTo(NamedTuple):
    start: Pos
    end: Pos

    def to_str(self) -> str:
        """ Return the 'str' version of `self` to be placed in a 'pos' field in the XML.
        """
        return self.start.to_str() + " " + self.end.to_str()

    @classmethod
    def from_str(cls, fromto: str) -> "FromTo":
        values = [float(v) for v in fromto.split()]
        assert len(values) == 6
        return cls(Pos(*values[:3]), Pos(*values[3:]))
    
    @classmethod
    def of_element(cls, element: Element, field: str = "fromto") -> "FromTo":
        if field not in element.attrib:
            raise RuntimeError(f"Element {element} doesn't have a '{field}' attribute.")
        return cls.from_str(element.attrib.get(field))

    def set_in_element(self, element: Element, field: str = "fromto") -> None:
        if field not in element.attrib:
            # NOTE: Refusing to set a new field for now.
            raise RuntimeError(f"Element {element} doesn't have a '{field}' attribute.")
        element.set(field, self.to_str())

    @property
    def center(self) -> Pos:
        return (self.start + self.end) / 2
