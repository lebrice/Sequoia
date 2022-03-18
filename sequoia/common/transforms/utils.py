from typing import Any

import numpy as np
from gym import spaces
from PIL import Image
from torch import Tensor

from sequoia.common.spaces.image import Image as ImageSpace


def is_image(v: Any) -> bool:
    """Returns wether the value is an Image, an image tensor, or an image
    space.
    """
    return (
        isinstance(v, Image.Image)
        or (isinstance(v, (Tensor, np.ndarray)) and len(v.shape) >= 3)
        or isinstance(v, ImageSpace)
        or isinstance(v, spaces.Box)
        and len(v.shape) >= 3
    )
