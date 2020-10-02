from dataclasses import dataclass, field
from typing import List

from simple_parsing import list_field
from utils.serialization import Serializable


@dataclass
class Task(Serializable):
    """ Dataclass that represents a task.

    TODO (@lebrice): This isn't being used anymore, but we could probably
    use it / add it to the Continuum package, if it doesn't already have something
    like it.
    TODO: Maybe the this could also specify from which dataset(s) it is sampled.
    """
    # The index of this task (the order in which it was encountered)
    index: int = field(default=-1, repr=False)
    # All the unique classes present within this task. (order matters)
    classes: List[int] = list_field()
