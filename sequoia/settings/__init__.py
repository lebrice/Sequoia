"""
"""
import inspect
from typing import Any, Dict, Iterable, List, Set, Type

from .base.objects import (Actions, ActionType, Observations,
                   ObservationType, Rewards, RewardType)
from .base.results import Results
from .base.environment import Environment
from .base.setting import Setting, SettingType
from .base.bases import SettingABC, Method
from .rl import *
from .sl import *

# # all concrete settings:
# all_settings: List[Type[Setting]] = [
#     ClassIncrementalSetting,
#     DomainIncrementalSetting,
#     TaskIncrementalSetting,
#     IIDSetting,
#     MultiTaskSetting,
#     ContinualRLSetting,
#     IncrementalRLSetting,
#     TaskIncrementalRLSetting,
#     RLSetting,
# ]
# Or, get All the settings:
all_settings: List[Type[Setting]] = frozenset([
    Setting, *Setting.all_children()
])
import inspect
print(inspect.getclasstree(all_settings, unique=True))
# assert False, all_settings