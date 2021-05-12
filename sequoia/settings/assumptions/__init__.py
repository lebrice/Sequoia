""" WIP: Mixin-style classes that define 'traits'/'assumptions' about a Setting.

IDEA: This package could define things that are to be reused in both the RL and 
the CL branches, kindof like a horizontal slice accross the tree.

The reasoning behind this is that some methods might require task labels, but
apply on both sides of the tree.
An alternative to this could also be to allow Methods to target multiple
settings, but this could get weird pretty quick.
"""
from .incremental import IncrementalSetting
# from .task_incremental import TaskIncrementalSetting
