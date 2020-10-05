"""
"""
import inspect
from typing import Any, Dict, Iterable, List, Set, Type

from .base import (Actions, ActionType, Environment, Observations,
                   ObservationType, Results, Rewards, RewardType, Setting,
                   SettingType)
from .active import *
from .passive import *

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
