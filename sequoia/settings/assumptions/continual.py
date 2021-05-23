from sequoia.settings.base import Setting
from sequoia.utils.utils import flag
from sequoia.utils import get_logger

from dataclasses import dataclass
from .base import AssumptionBase
logger = get_logger(__file__)


@dataclass
class ContinualAssumption(AssumptionBase):
    """ Assumptions for Setting where the environments change over time. """
    known_task_boundaries_at_train_time: bool = flag(False)
    # Wether we get informed when reaching the boundary between two tasks during
    # training. Only used when `smooth_task_boundaries` is False.
    known_task_boundaries_at_test_time: bool = flag(False)
    # Wether we have sudden changes in the environments, or if the transition
    # are "smooth".
    smooth_task_boundaries: bool = flag(True)

    # TODO: Move everything necessary to get ContinualRLSetting to work out of
    # Incremental and into this here. Makes no sense that ContinualRLSetting inherits
    # from Incremental, rather than this!
