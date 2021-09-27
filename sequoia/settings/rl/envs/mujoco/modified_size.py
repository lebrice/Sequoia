import hashlib
import inspect
import os
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import ClassVar, Dict, List
from logging import getLogger as get_logger
from gym.envs.mujoco import MujocoEnv

logger = get_logger(__name__)


def change_size_in_xml(
    tree: ET.ElementTree, **body_name_to_size_scale: Dict[str, float]
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


def get_geom_sizes(tree: ET.ElementTree, body_name: str) -> List[float]:
    # body = tree.find(f".//body[@name='{body_name}']")
    geom = tree.find(f".//geom[@name='{body_name}']")
    if geom is None:
        geom = tree.find(f".//geom[@name='{body_name}_geom']")
    assert geom is not None
    assert "size" in geom.attrib
    # print(body_name)
    # print("Old size: ", geom.attrib["size"])
    sizes: List[float] = [float(s) for s in geom.attrib["size"].split(" ")]
    return sizes


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

        if model_path.startswith("/"):
            full_path = model_path
        else:
            full_path = os.path.join(
                os.path.dirname(inspect.getsourcefile(MujocoEnv)), "assets", model_path
            )
        if not os.path.exists(full_path):
            raise IOError(f"File {full_path} does not exist")

        # find the body_part we want

        if any(scale_factor == 0 for scale_factor in size_scales):
            raise RuntimeError("Can't use a scale_factor of 0!")

        logger.debug(f"Default XML path: {full_path}")
        self.default_tree = ET.parse(full_path)
        self.tree = self.default_tree

        if body_name_to_size_scale:
            logger.info(f"Changing parts: {body_name_to_size_scale}")
            self.tree = change_size_in_xml(self.default_tree, **body_name_to_size_scale)
            # create new xml
            # IDEA: Create an XML file with a unique name somewhere, and then write the
            hash_str = hashlib.md5(
                (str(self) + str(body_name_to_size_scale)).encode()
            ).hexdigest()
            temp_dir = Path(tempfile.gettempdir())
            new_xml_path = temp_dir / f"{hash_str}.xml"
            if not new_xml_path.parent.exists():
                new_xml_path.parent.mkdir(exist_ok=False, parents=True)
            self.tree.write(str(new_xml_path))
            logger.info(f"Generated XML path: {new_xml_path}")

            # Update the value to be passed to the constructor:
            full_path = str(new_xml_path)

        self.body_name_to_size_scale = body_name_to_size_scale
        # load the modified xml
        super().__init__(model_path=full_path, frame_skip=frame_skip, **kwargs)
