""" Task Incremental setting / experiment / methods.

Invoke like so:
"""
# 1. Import stuff from the Parent
# NOTE: Here there doesn't seem to be a need for a custom 'Results' class for
# TaskIncremental, given how similar it is to ClassIncremental.
from .. import ClassIncrementalResults as TaskIncrementalResults

# 2. Import what we overwrite/customize
from .task_incremental_setting import TaskIncrementalSetting
from .iid import *
from .multi_task import *
