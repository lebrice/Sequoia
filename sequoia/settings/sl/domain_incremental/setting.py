from dataclasses import dataclass

from sequoia.settings.sl.incremental.setting import IncrementalSLSetting
from sequoia.utils.utils import constant


@dataclass
class DomainIncrementalSLSetting(IncrementalSLSetting):
    """Supervised CL Setting where the input domain shifts incrementally.

    Task labels and task boundaries are given at training time, but not at test-time.
    The crucial difference between the Domain-Incremental and Class-Incremental settings
    is that the action space is smaller in domain-incremental learning, as it is a
    `Discrete(n_classes_per_task)`, rather than the `Discrete(total_classes)` in
    Class-Incremental setting.

    For example: Create a classifier for odd vs even hand-written digits. It first be
    trained on digits 0 and 1, then digits 2 and 3, then digits 4 and 5, etc.
    At evaluation time, it will be evaluated on all digits
    """

    shared_action_space: bool = constant(True)
