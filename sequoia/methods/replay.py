from .replay_wrapper import ReplayEnvWrapper

from sequoia.methods.base_method import BaseMethod, BaseModel
from sequoia.settings.sl import SLSetting, SLEnvironment
from dataclasses import dataclass

from sequoia.common.config import Config
from sequoia.methods.trainer import TrainerConfig


@dataclass
class HParams(BaseModel.HParams):
    replay_buffer_capacity: int = 1000
    replay_sample_size: int = 1000


@dataclass
class Replay(BaseMethod, target_setting=SLSetting):
    hparams: HParams = HParams()

    def __init__(
        self,
        hparams: HParams = None,
        config: Config = None,
        trainer_options: TrainerConfig = None,
        **kwargs
    ):
        super().__init__(
            hparams=hparams, config=config, trainer_options=trainer_options, **kwargs
        )
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
                capacity=self.hparams.replay_buffer_capacity,
                sample_size=self.hparams.replay_sample_size,
            )
            train_env = self.replay_wrapper
        else:
            task_id = self.current_task_id
            # Pass the buffer to the new wrapper.
            self.replay_wrapper = self.replay_wrapper.for_next_env(train_env, task_id=task_id)
        self.replay_wrapper.enable_collection
