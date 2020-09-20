
# from .active import *
from typing import Any, Dict, Iterable, List, Set, Type

from utils.utils import get_all_concrete_subclasses_of, get_all_subclasses_of
import inspect
from .base import *
from .passive import *
from .active import *

all_settings: List[Type[Setting]] = [
    IIDSetting,
    TaskIncrementalSetting,
    ClassIncrementalSetting,
    ContinualRLSetting,
    ClassIncrementalRLSetting,
    TaskIncrementalRLSetting,
    RLSetting,
    # setting for name, setting in vars().items() if inspect.isclass(setting) and issubclass(setting, Setting)
]
