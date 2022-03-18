"""
"""
import inspect
from typing import Any, Dict, Iterable, List, Set, Type

from .base.bases import Method, SettingABC
from .base.environment import Environment
from .base.objects import Actions, ActionType, Observations, ObservationType, Rewards, RewardType
from .base.results import Results
from .base.setting import Setting, SettingType
from .rl import *
from .sl import *

# # all concrete settings:
# all_settings: List[Type[Setting]] = [
#     ClassIncrementalSetting,
#     DomainIncrementalSetting,
#     TaskIncrementalSLSetting,
#     TraditionalSLSetting,
#     MultiTaskSetting,
#     ContinualRLSetting,
#     IncrementalRLSetting,
#     TaskIncrementalRLSetting,
#     RLSetting,
# ]
# Or, get All the settings:
all_settings: Set[Type[SettingABC]] = set([Setting, *Setting.children()])
# FIXME: Remove this, just checking the inspect atm.:
# import inspect
# import pprint

# print(Setting.get_tree_string())
# exit()

# print(inspect.getclasstree(all_settings, unique=True))
# assert False
# assert False, all_settings
