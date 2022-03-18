""" This module defines the `BaseModel` used by the `BaseMethod`.

Output heads are available for both Supervised and Reinforcement Learning, and can be
found in `sequoia.methods.models.output_heads`.

Instead of defining the `Model` in one large file, it is instead split into a base
class (`Model`, defined in `model.py`) on top of which a few "mixins" are added, each
of which adds additional functionality:

- [SemiSupervisedModel](self_supervised_model.py):
    Adds support for semi-supervised (partially labeled or un-labeled) training, by
    splitting up partially labeled batches into a fully labeled sub-batch and a fully
    unlabeled sub-batch.

- [MultiHeadModel](multihead_model.py):
    Adds support for:
    - multi-head prediction: Using a dedicated output head for each task when
      task labels are available
    - Mixed batches (data coming from more than one task within the same batch)
    - TODO: Task inference: When task labels aren't available, perform
      some task inference in order to choose which output head to use.

- [SelfSupervisedModel](self_supervised_model.py):
    Adds methods for adding self-supervised losses to the model using different
    Auxiliary Tasks.
    
The `BaseModel` is then formed by inheriting from each of these mixins.
"""
from .base_model import BaseModel

# TODO: Maybe the naming of these could be a bit better: Model seems more 'general' than BaseModel.
from .model import Model, available_encoders, available_optimizers
from .multihead_model import MultiHeadModel
from .self_supervised_model import SelfSupervisedModel
from .semi_supervised_model import SemiSupervisedModel
