from .environment import RLEnvironment
from .setting import RLSetting

ActiveEnvironment = RLEnvironment
from .continual import ContinualRLSetting, make_continuous_task
from .discrete import DiscreteTaskAgnosticRLSetting, make_discrete_task
from .incremental import IncrementalRLSetting, make_incremental_task

# TODO: Properly Add the multi-task RL setting.
from .multi_task import MultiTaskRLSetting
from .task_incremental import TaskIncrementalRLSetting
from .traditional import TraditionalRLSetting
