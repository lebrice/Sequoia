""" Task Incremental Setting 

Adds the additional assumption that the task labels are available at test time.
"""
# 1. Import stuff from the Parent
# NOTE: Here there doesn't seem to be a need for a custom 'Results' class for
# TaskIncremental, given how similar it is to ClassIncremental.
# 2. Import what we overwrite/customize
from .setting import TaskIncrementalSLSetting
