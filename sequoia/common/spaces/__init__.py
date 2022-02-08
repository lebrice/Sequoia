""" Custom `gym.spaces.Space` subclasses used by Sequoia. """
from .image import Image, ImageTensorSpace
from .named_tuple import NamedTuple, NamedTupleSpace
from .space import Space
from .sparse import Sparse
from .tensor_spaces import TensorBox, TensorDiscrete, TensorMultiDiscrete, TensorSpace
from .typed_dict import TypedDictSpace
