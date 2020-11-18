""" TODO: Implement and test PPO. """
from dataclasses import dataclass
from typing import ClassVar, Type
from simple_parsing import mutable_field

from stable_baselines3.ppo import PPO

from methods import register_method

from .base import StableBaselines3Method, BaseAlgorithm, SB3BaseHParams


class PPOModel(PPO):
    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters of the PPO Model. """
        # TODO: Create the fields from the PPO constructor arguments.
        pass


@register_method
@dataclass
class PPOMethod(StableBaselines3Method):
    """ Method that uses the PPO model from stable-baselines3. """
    Model: ClassVar[Type[PPOModel]] = PPOModel
    # Hyper-parameters of the PPO Model.
    hparams: PPOModel.HParams = mutable_field(PPOModel.HParams)
