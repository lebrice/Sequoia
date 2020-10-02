"""
"""
from typing import Any, Dict, Iterable, List, Set, Type
import inspect

from .base import (ActionType, EnvironmentBase, ObservationType, Results,
                   RewardType, Setting, SettingType)
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
    ## OR, dynamic version:
    # setting for name, setting in vars().items()
    # if inspect.isclass(setting) and issubclass(setting, Setting)
]
