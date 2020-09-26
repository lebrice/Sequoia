
# from .active import *
from typing import Any, Dict, Iterable, List, Set, Type
import inspect

from .base import *
from .passive import *
from .active import *
from .pl_bolts_settings.pl_bolt_setting import MnistSetting
all_settings: List[Type[Setting]] = [
    IIDSetting,
    TaskIncrementalSetting,
    ClassIncrementalSetting,
    ContinualRLSetting,
    ClassIncrementalRLSetting,
    TaskIncrementalRLSetting,
    RLSetting,
    MnistSetting,
    ## OR, dynamic version:
    # setting for name, setting in vars().items()
    # if inspect.isclass(setting) and issubclass(setting, Setting)
]
