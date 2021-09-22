from sequoia.conftest import mujoco_required

pytestmark = mujoco_required
import itertools

from .hopper import ContinualHopperV2Env, ContinualHopperV3Env
from .modified_gravity_test import ModifiedGravityEnvTests
from .modified_size_test import ModifiedSizeEnvTests
from .modified_mass_test import ModifiedMassEnvTests
from typing import ClassVar, Type

import pytest
import os
import inspect
from gym.envs.mujoco import MujocoEnv
from xml.etree.ElementTree import ElementTree, Element, parse, fromstring, tostring
from pathlib import Path


from sequoia.conftest import mujoco_required

# # TODO: There is a bug in the way the hopper XML is generated, where the sticks / joints don't seem to follow.
# bob = ContinualHopperEnv(body_name_to_size_scale={"thigh": 2})
# assert False, bob


@mujoco_required
class TestContinualHopperV2Env(
    ModifiedGravityEnvTests, ModifiedSizeEnvTests, ModifiedMassEnvTests
):
    Environment: ClassVar[Type[ContinualHopperV2Env]] = ContinualHopperV2Env


@mujoco_required
class TestContinualHopperV3Env(
    ModifiedGravityEnvTests, ModifiedSizeEnvTests, ModifiedMassEnvTests
):
    Environment: ClassVar[Type[ContinualHopperV3Env]] = ContinualHopperV3Env


def load_tree(model_path: Path) -> ElementTree:
    # model_path = "hopper.xml"
    if model_path.startswith("/"):
        full_path = model_path
    else:
        full_path = os.path.join(
            os.path.dirname(inspect.getsourcefile(MujocoEnv)), "assets", model_path
        )
    if not os.path.exists(full_path):
        raise IOError(f"File {full_path} does not exist")

    with open(model_path, "r") as f:
        return f.read()


default_hopper_body_xml = f"""\
<worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1" />
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane" />
    <body name="torso" pos="0 0 1.25">
        <camera name="track" mode="trackcom" pos="0 -3 1" xyaxes="1 0 0 0 0 1" />
        <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide" />
        <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" ref="1.25" stiffness="0" type="slide" />
        <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 1.25" stiffness="0" type="hinge" />
        <geom friction="0.9" fromto="0 0 1.45 0 0 1.05" name="torso_geom" size="0.05" type="capsule" />
        <body name="thigh" pos="0 0 1.05">
            <joint axis="0 -1 0" name="thigh_joint" pos="0 0 1.05" range="-150 0" type="hinge" />
            <geom friction="0.9" fromto="0 0 1.05 0 0 0.6" name="thigh_geom" size="0.05" type="capsule" />
            <body name="leg" pos="0 0 0.35">
                <joint axis="0 -1 0" name="leg_joint" pos="0 0 0.6" range="-150 0" type="hinge" />
                <geom friction="0.9" fromto="0 0 0.6 0 0 0.1" name="leg_geom" size="0.04" type="capsule" />
                <body name="foot" pos="0.13/2 0 0.1">
                    <joint axis="0 -1 0" name="foot_joint" pos="0 0 0.1" range="-45 45" type="hinge" />
                    <geom friction="2.0" fromto="-0.13 0 0.1 0.26 0 0.1" name="foot_geom" size="0.06" type="capsule" />
                </body>
            </body>
        </body>
    </body>
</worldbody>
"""


def elements_equal(e1, e2) -> bool:
    """Taken from https://stackoverflow.com/a/24349916/6388696"""
    assert e1.tag == e2.tag
    assert e1.text == e2.text
    assert e1.tail == e2.tail
    assert e1.attrib == e2.attrib
    assert len(e1) == len(e2)
    assert all(elements_equal(c1, c2) for c1, c2 in zip(e1, e2))


@pytest.mark.xfail(reason="Dropping this for now, XML is really annoying.")
@pytest.mark.parametrize(
    "input_xml_str, scale_factor, output_xml_str",
    [
        (
            default_hopper_body_xml,
            1.0,
            default_hopper_body_xml,
        ),
        (
            default_hopper_body_xml,
            2.0,
            f"""\
        <worldbody>
            <body name="torso" pos="0 0 {1.45}">
                <camera name="track" mode="trackcom" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
                <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
                <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" ref="{1.25}" stiffness="0" type="slide"/>
                <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 {1.45}" stiffness="0" type="hinge"/>
                <geom friction="0.9" fromto="0 0 {1.85} 0 0 1.05" name="torso_geom" size="{0.10}" type="capsule"/>
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
        """,
        ),
    ],
    ids=(f"param{i}" for i in itertools.count()),
)
def test_change_torso(input_xml_str: str, scale_factor: float, output_xml_str: str):

    # # TODO: Get rid of annoying whitespace issues!
    from xml.etree.ElementTree import XMLParser

    input_tree = fromstring(input_xml_str)
    expected = fromstring(output_xml_str)

    # from io import StringIO
    # in_file = StringIO(input_xml_str)
    # out_file = StringIO(output_xml_str)
    # input_tree = parse(in_file)
    # expected = parse(out_file)

    update_torso(tree=input_tree, size_scale_factor=scale_factor)
    # import textwrap
    # from xml.dom import minidom
    # result = minidom.parseString(tostring(input_tree, method="text")).toprettyxml()
    result = input_tree
    assert elements_equal(result, expected)
    # expected = minidom.parseString().toprettyxml()
    assert result == expected
