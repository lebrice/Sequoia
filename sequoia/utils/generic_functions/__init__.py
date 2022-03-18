""" Defines a bunch of single-dispatch generic functions, that are applicable
on structured objects, numpy arrays, tensors, spaces, etc.
"""
from ._namedtuple import NamedTuple, is_namedtuple
from .concatenate import concatenate
from .detach import detach
from .move import move
from .replace import replace
from .singledispatchmethod import singledispatchmethod
from .slicing import get_slice, set_slice
from .stack import stack
from .to_from_tensor import from_tensor, to_tensor
