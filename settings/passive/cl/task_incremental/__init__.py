""" Task Incremental setting / experiment / methods.

Invoke like so:
"""
# 1. Import stuff from the Parent
# NOTE:  (no need here since we overwrite basically everything from the base experiment.)


# 2. Import what we overwrite/customize
from .task_incremental_setting import TaskIncrementalSetting
from .results import TaskIncrementalResults
from .iid import *
