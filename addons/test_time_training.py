from dataclasses import dataclass

from torch import Tensor

from common.losses import LossInfo
from experiment import ExperimentBase


@dataclass  # type: ignore
class TestTimeTrainingAddon(ExperimentBase):
    """ Experiment where we also perform self-supervised training at test-time.
    """
    # Wether or not to train using self-supervision even at test-time.
    test_time_training: bool = False

    def test_batch(self, data: Tensor, target: Tensor=None) -> LossInfo:
        if self.test_time_training:
            super().train_batch(data, None)
        return super().test_batch(data, target)
