""" Miscelaneous utility functions. """
import sys
from .utils import *
from .serialization import Serializable
from .logging_utils import get_logger
from .parseable import Parseable

from .generic_functions import *

if sys.version_info >= (3, 8):
    from functools import singledispatchmethod  # type: ignore
else:
    try:
        from singledispatchmethod import singledispatchmethod
    except ImportError as e:
        print(f"Couldn't import singledispatchmethod: {e}")
        print("Since you're running python version below 3.8, you need to "
              "install the backport for singledispatchmethod (which was added "
              "to functools in python 3.8), using the following command:\n"
              "> pip install singledispatchmethod")
        exit()