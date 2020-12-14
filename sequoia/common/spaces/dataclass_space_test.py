

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .dataclass_space import allow_dataclasses_as_dict_samples



from gym import Space, spaces

@dataclass
class Sample:
    a: np.ndarray
    b: bool
    c: Tuple[int, int]



def test_supports_dataclasses():
    # IDEA: Wrapper that makes the 'default factory' of each field actually use
    # the 'sample' method from a space associated with each class.
    
    space = spaces.Dict(
        a=spaces.Box(0, 1, [2, 2]),
        b=spaces.Box(False, True, [1], np.bool),
        c=spaces.MultiDiscrete([2, 2])
    )

    wrapped_space: space.Dict = allow_dataclasses_as_dict_samples(space, Sample)
    assert isinstance(wrapped_space, spaces.Dict)
    assert Sample(
        a=np.ones([2, 2]),
        b=[False],
        c=np.array([0, 1]),
    ) in wrapped_space
    assert isinstance(wrapped_space.sample(), Sample)
