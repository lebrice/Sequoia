from sequoia.settings.base import Setting
from sequoia.utils.utils import flag

from dataclasses import dataclass

@dataclass
class ContinualSetting(Setting):
    """ Assumptions for Setting where the environments change over time. """
    known_task_boundaries_at_train_time: bool = flag(False)
    # Wether we get informed when reaching the boundary between two tasks during
    # training. Only used when `smooth_task_boundaries` is False.
    known_task_boundaries_at_test_time: bool = flag(False)
    # Wether we have sudden changes in the environments, or if the transition
    # are "smooth".
    smooth_task_boundaries: bool = flag(True)
