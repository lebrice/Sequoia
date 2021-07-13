from .setting import RLSetting
from .environment import ActiveEnvironment
from .continual import ContinualRLSetting, make_continuous_task
from .discrete import DiscreteTaskAgnosticRLSetting, make_discrete_task
from .incremental import IncrementalRLSetting, make_incremental_task
from .task_incremental import TaskIncrementalRLSetting
# TODO: Properly Add the multi-task RL setting.
from .multi_task import MultiTaskRLSetting
from .traditional import TraditionalRLSetting
