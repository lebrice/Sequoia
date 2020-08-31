from typing import NamedTuple

class Dims(NamedTuple):
    """ A small little namedtuple to help with dimensions of tensors.

    TODO: Use the named dimensions in pytorch instead.
    """
    h: int
    w: int
    c: int
