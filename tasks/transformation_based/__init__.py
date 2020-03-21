from .adjust_brightness import AdjustBrightnessTask
from .bases import (ClassifyTransformationTask, RegressTransformationTask,
                    TransformationBasedTask)
from .rotation import RotationTask

__all__ = [
    "AdjustBrightnessTask",
    "ClassifyTransformationTask", "RegressTransformationTask",
    "TransformationTask",
    "RotationTask",
]
