""" Defines a bunch of single-dispatch generic functions, that are applicable
on structured objects, numpy arrays, tensors, spaces, etc.
"""
from ._namedtuple import NamedTuple, is_namedtuple
from .move import move
from .detach import detach
from .replace import replace
from .slicing import get_slice, set_slice
from .stack import stack