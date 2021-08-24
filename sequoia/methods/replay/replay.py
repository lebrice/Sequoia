from typing import Optional
from sequoia.methods.base_method import BaseMethod, BaseModel
from sequoia.settings.sl import SLSetting, SLEnvironment
from dataclasses import dataclass

from sequoia.common.config import Config
from sequoia.methods.trainer import TrainerConfig
from simple_parsing.helpers.hparams import HyperParameters, uniform 
from simple_parsing.helpers import mutable_field 
from .wrapper import ReplayEnvWrapper


@dataclass
class ReplayHParams(HyperParameters):
    """ Hyper-parameters specific to the Replay method. """
    # Maximum size of the replay buffer.
    replay_buffer_capacity: int = uniform(100, 10_000, default=1000)
    # Number of replay samples to add to each batch.
    replay_sample_size: int = uniform(10, 128, default=32)


@dataclass
class Replay(BaseMethod, target_setting=SLSetting):
    replay_params: ReplayHParams = mutable_field(ReplayHParams)

    def __init__(
        self,
        hparams: BaseModel.HParams = None,
        config: Config = None,
        trainer_options: TrainerConfig = None,
        replay_hparams: ReplayHParams = None,
        **kwargs
    ):
        super().__init__(
            hparams=hparams, config=config, trainer_options=trainer_options, **kwargs
        )
        self.replay_hparams = replay_hparams or ReplayHParams()
        self.replay_wrapper: Optional[ReplayEnvWrapper] = None

    def configure(self, setting: SLSetting):
        super().configure(setting)
        self.replay_wrapper = None

    def on_task_switch(self, task_id):
        self.current_task_id = task_id
        return super().on_task_switch(task_id)

    def fit(self, train_env: SLEnvironment, valid_env: SLEnvironment):
        if self.replay_wrapper is None:
            self.replay_wrapper = ReplayEnvWrapper(
                train_env,
                capacity=self.replay_hparams.replay_buffer_capacity,
                sample_size=self.replay_hparams.replay_sample_size,
                device=self.config.device,
            )
        else:
            # NOTE: Also works in the case where `task_id` is None. In that case, the
            # observations and rewards will be sampled from all previous batches, rather
            # than from the batches from other tasks.
            task_id = self.current_task_id
            # Pass the buffer to the new wrapper.
            self.replay_wrapper = self.replay_wrapper.for_next_env(
                train_env, task_id=task_id
            )
        train_env = self.replay_wrapper
        super().fit(train_env=train_env, valid_env=valid_env)


if __name__ == "__main__":
    Replay.main()
