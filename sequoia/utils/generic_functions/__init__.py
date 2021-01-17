""" Defines a bunch of single-dispatch generic functions, that are applicable
on structured objects, numpy arrays, tensors, spaces, etc.
"""
from .singledispatchmethod import singledispatchmethod
from ._namedtuple import NamedTuple, is_namedtuple
from .move import move
from .detach import detach
from .replace import replace
from .slicing import get_slice, set_slice
from .stack import stack
from .concatenate import concatenate
from .to_from_tensor import to_tensor, from_tensor
