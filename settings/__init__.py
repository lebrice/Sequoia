"""
"""
import inspect
from typing import Any, Dict, Iterable, List, Set, Type
# from .setting_abc import SettingABC
# from .method_abc import MethodABC
from .base import (Actions, ActionType, Environment, Observations,
                   ObservationType, Results, Rewards, RewardType, Setting,
                   SettingType)
from .active import *
from .passive import *
# all_settings: List[Type[Setting]] = [
#     IIDSetting,
#     TaskIncrementalSetting,
#     ClassIncrementalSetting,
#     ContinualRLSetting,
#     ClassIncrementalRLSetting,
#     TaskIncrementalRLSetting,
#     RLSetting,
# ]
## OR, dynamic version:
all_settings: List[Type[Setting]] = frozenset([
    Setting, *Setting.all_children()
])
