from ..incremental.tasks import make_incremental_task, sequoia_registry, task_sampling_function
from functools import singledispatch
# NOTE: For now there aren't any tasks specific to only task-incremental.
make_task_incremental_task = make_incremental_task
is_supported = make_task_incremental_task.is_supported
