from .. import Results
from .environment import PassiveEnvironment
from .setting import SLSetting
from .continual import ContinualSLSetting
from .incremental import IncrementalSLSetting
ClassIncrementalSetting = IncrementalSLSetting
# from .class_incremental import ClassIncrementalSetting
from .task_incremental import TaskIncrementalSLSetting
from .domain_incremental import DomainIncrementalSLSetting
from .multi_task import MultiTaskSLSetting
from .traditional import TraditionalSLSetting

# TODO: Import variants without the 'SL' in it above, and then don't include then in the
# __all__ below, to improve backward compatibility a bit.
# __all__ = [
#     "PassiveEnvironment",
#     "SLSetting", ...
# ]
