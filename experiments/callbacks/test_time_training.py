"""TODO: Migrate to Pytorch-Lightning.

Idea: This could maybe be an extension for Classifier? (Extending Classifier)?

"""

# from dataclasses import dataclass

# from torch import Tensor

# from common.losses import LossInfo
# from .addon import ExperimentAddon
# from simple_parsing import mutable_field


# @dataclass  # type: ignore
# class TestTimeTrainingAddon(ExperimentAddon):
#     """ Experiment where we also perform self-supervised training at test-time.
#     """
#     @dataclass
#     class Config(ExperimentAddon.Config):
#         # Wether or not to train using self-supervision even at test-time.
#         test_time_training: bool = False
    
#     config: Config = mutable_field(Config)

#     def test_batch(self, data: Tensor, target: Tensor=None, **kwargs) -> LossInfo:  # type: ignore
#         if self.config.test_time_training:
#             super().train_batch(data, None, **kwargs)  # type: ignore
#         return super().test_batch(data, target, **kwargs)  # type: ignore
