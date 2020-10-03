""" WIP: These modules are subclasses of Model that add optional functionality.

TODO: We could either renamed `Model` to `BaseModel` add them all on top of
`BaseModel` to create the finalized `Model` class, a bit like what we did
earlier with addons, and Pytorch-Lightning does for their Trainer class.
"""
from .class_incremental_model import ClassIncrementalModel
from .self_supervised_model import SelfSupervisedModel
from .semi_supervised_model import SemiSupervisedModel
# from .task_incremental_model import TaskIncrementalModel