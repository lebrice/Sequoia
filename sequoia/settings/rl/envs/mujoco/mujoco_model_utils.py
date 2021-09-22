from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np
from typing import NamedTuple, Any
from xml.etree.ElementTree import Element
from typing import Sequence, Tuple, Union

import numpy as np


def pos_to_str(pos: Tuple[float, ...]) -> str:
    return " ".join("0" if v == 0 else str(round(v, 5)) for v in pos)


def str_to_pos(pos_str: str) -> "Pos":
    return Pos(*[float(v) for v in pos_str.split()])


class Pos(NamedTuple):
    x: float
    y: float
    z: float

    def to_str(self) -> str:
        """Return the 'str' version of `self` to be placed in a 'pos' field in the XML."""
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
        """Return the 'str' version of `self` to be placed in a 'pos' field in the XML."""
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


import textwrap


@dataclass
class FromTo:
    from_x: float
    from_y: float
    from_z: float
    to_x: float
    to_y: float
    to_z: float

    def __str__(self):
        return " ".join(
            [self.from_x, self.from_y, self.from_z, self.to_x, self.to_y, self.to_z]
        )


from dataclasses import dataclass


@dataclass
class TorsoGeom:
    friction: float = 0.9
    fromto = FromTo(0, 0, 1.45, 0, 0, 1.05)
    name: str = "torso_geom"
    size: float = 0.05
    type: str = "capsule"

    def render_xml(self) -> str:
        return f"""<geom friction="{self.friction}" fromto="{self.fromto}" name="{self.name}" size="{self.size}" type="{self.type}"/>"""


@dataclass
class HoperV3Model:
    torso_geom: TorsoGeom

    def render_xml(self) -> str:
        return textwrap.dedent(
            """\
            <mujoco model="hopper">
            <compiler angle="degree" coordinate="global" inertiafromgeom="true"/>
            <default>
                <joint armature="1" damping="1" limited="true"/>
                <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1" solimp=".8 .8 .01" solref=".02 1"/>
                <motor ctrllimited="true" ctrlrange="-.4 .4"/>
            </default>
            <option integrator="RK4" timestep="0.002"/>
            <visual>
                <map znear="0.02"/>
            </visual>
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
            <actuator>
                <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="thigh_joint"/>
                <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="leg_joint"/>
                <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="foot_joint"/>
            </actuator>
                <asset>
                    <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0"
                        width="100" height="100"/>
                    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
                    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
                    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
                    <material name="geom" texture="texgeom" texuniform="true"/>
                </asset>
            </mujoco>
            """
        )
