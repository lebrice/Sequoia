from copy import deepcopy
import inspect
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import ClassVar, Dict, List

import numpy as np
from gym.envs.mujoco import MujocoEnv


def change_size_in_xml(
    tree: ET.ElementTree,
    **body_name_to_size_scale: Dict[str, float]
) -> ET.ElementTree:
    tree = deepcopy(tree)
    for body_name, size_scale in body_name_to_size_scale.items():
        body = tree.find(f".//body[@name='{body_name}']")
        geom = tree.find(f".//geom[@name='{body_name}']")
        if geom is None:
            geom = tree.find(f".//geom[@name='{body_name}_geom']")
        assert geom is not None
        assert "size" in geom.attrib
        # print(body_name)
        # print("Old size: ", geom.attrib["size"])
        sizes: List[float] = [float(s) for s in geom.attrib["size"].split(" ")]
        new_sizes = [size * size_scale for size in sizes]
        geom.attrib["size"] = " ".join(map(str, new_sizes))
        # print("New size: ", geom.attrib['size'])
    return tree


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
            print(f"Changing parts: {body_name_to_size_scale}")
            # NOTE: For now this still modifies `tree` in-place.
            tree = change_size_in_xml(tree, **body_name_to_size_scale)
            # create new xml
            _, new_xml_path = tempfile.mkstemp(suffix=".xml", text=True)
            tree.write(new_xml_path)
            print(f"Generated XML path: {new_xml_path}")
            full_path = new_xml_path
        self.body_name_to_size_scale = body_name_to_size_scale
        # load the modified xml
        super().__init__(model_path=full_path, frame_skip=frame_skip, **kwargs)
