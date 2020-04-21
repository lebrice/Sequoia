from dataclasses import dataclass

import torch
from torch import Tensor

from common.losses import LossInfo
from task_incremental import TaskIncremental
from experiment import Experiment
from iid import IID
from contextlib import nullcontext
from tasks.tasks import Tasks


@dataclass  # type: ignore
class TestTimeTraining(Experiment):
    """ Experiment where we also perform self-supervised training at test-time.
    """
    # Wether or not to train using self-supervision at test-time.
    test_time_training: bool = False

    def train_batch_self_supervised(self, data: Tensor, target: Tensor) -> LossInfo:
        self.model.optimizer.zero_grad()

        batch_loss_info = self.model.get_loss(data, target)
        total_loss = batch_loss_info.total_loss
        # print("All the tasks:", batch_loss_info.losses.keys())
        
        self_supervised_loss = torch.zeros(1).to(self.config.device)
        for name, loss_info in batch_loss_info.losses.items():
            if name != Tasks.SUPERVISED:
                self_supervised_loss += loss_info.total_loss
        self_supervised_loss.backward()
        
        self.model.optimizer.step()
        return batch_loss_info

    def test_batch(self, data: Tensor, target: Tensor) -> LossInfo:
        if self.test_time_training:
            return self.train_batch_self_supervised(data, target)
        else:
            return super().test_batch(data, target)
@dataclass
class IIDWithTestTimeTraining(IID, TestTimeTraining):
    test_time_training: bool = True

@dataclass
class TaskIncrementalWithTestTimeTraining(TaskIncremental, TestTimeTraining):
    test_time_training: bool = True

from simple_parsing import subparsers

@dataclass
class TestTimeTrainingOptions:
    # Which experiment to run with test-time training.
    experiment: TestTimeTraining = subparsers({
        "iid": IIDWithTestTimeTraining,
        "task-incremental": TaskIncrementalWithTestTimeTraining,
    })

    def __post_init__(self):
        from main import launch
        launch(self.experiment)
        exit()


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    import textwrap
    parser = ArgumentParser()
    parser.add_arguments(TestTimeTrainingOptions, "options")
    args = parser.parse_args()
