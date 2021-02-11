""" This module defines the BaselineModel.

The BaselineModel is based on the BaseModel (from base_model.py), on top of
which the following 'addons' get added:

- [SemiSupervisedModel](self_supervised_model.py):
    Adds support for semi-supervised (partially labeled or un-labeled) batches of data.

- [MultiHeadModel](multihead_model.py):
    Adds support for:
    - multi-head prediction: Using a dedicated output head for each task when
      task labels are available
    - Mixed batches (data coming from more than one task within the same batch)
    - TODO: Task inference: When task labels aren't available, perform
      some task inference in order to choose which output head to use.

- SelfSupervisedModel:
    Adds methods for adding self-supervised losses to the model using different
    Auxiliary Tasks.
"""
from .base_hparams import BaseHParams, available_encoders, available_optimizers
from .base_model import BaseModel
from .multihead_model import MultiHeadModel
from .self_supervised_model import SelfSupervisedModel
from .semi_supervised_model import SemiSupervisedModel
from .baseline_model import BaselineModel