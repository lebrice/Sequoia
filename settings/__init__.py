"""
"""
import inspect
from typing import Any, Dict, Iterable, List, Set, Type
# from .setting_base import SettingABC
# from .method_abc import MethodABC

from .base.objects import (Actions, ActionType, Observations,
                   ObservationType, Rewards, RewardType)
from .base.results import Results
from .base.environment import Environment
from .base.setting import Setting, SettingType
from .active import *
from .passive import *
# all concrete settings:
all_settings: List[Type[Setting]] = [
    IIDSetting,
    TaskIncrementalSetting,
    ClassIncrementalSetting,
    ContinualRLSetting,
    ClassIncrementalRLSetting,
    TaskIncrementalRLSetting,
    RLSetting,
]
## Or, get All the settings:
# all_settings: List[Type[Setting]] = frozenset([
#     Setting, *Setting.all_children()
# ])
