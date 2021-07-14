""" Custom `gym.spaces.Space` subclasses used by Sequoia. """
from .space import Space
from .sparse import Sparse
from .image import Image
from .named_tuple import NamedTuple, NamedTupleSpace
from .typed_dict import TypedDictSpace
from .tensor_spaces import TensorSpace, TensorBox, TensorDiscrete, TensorMultiDiscrete
