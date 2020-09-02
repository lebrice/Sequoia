""" [WIP] Example of a method targetting an RL setting.

"""
from dataclasses import dataclass
from typing import List, Optional, Type, Union, Iterable

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.trainer.data_connector import DataConnector

from common.config import TrainerConfig
from settings import RLSetting
from simple_parsing import mutable_field

from .method import Method
from .models.agent import Agent


@dataclass
class RLTrainerConfig(TrainerConfig):
    val_check_interval: Union[int, float] = 100


@dataclass
class RLMethod(Method, target_setting=RLSetting):
    """ Method aimed at solving an RL setting. """

    # Options for the Trainer object.
    trainer_options: RLTrainerConfig = mutable_field(RLTrainerConfig)


    def model_class(self, setting: RLSetting) -> Type[Agent]:
        """Retuns the type of

        Args:
            setting (RLSetting): [description]

        Returns:
            Type[Agent]: [description]
        """
        return Agent


if __name__ == "__main__":
    RLMethod.main()
