""" [WIP] Example of a method targetting an RL setting.

"""
from dataclasses import dataclass
from typing import Iterable, List, Optional, Type, Union

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from common.config import TrainerConfig
from settings import RLSetting
from simple_parsing import mutable_field

from .method import Method
from .models.actor_critic_agent import ActorCritic
from .models.agent import Agent


@dataclass
class RLTrainerConfig(TrainerConfig):
    """ Config for the Trainer object used by pytorch lightning. 
    """
    # NOTE: The default value for the validation interval is changed compared to
    # the default value of the parent, because the 'GymDatasets' are
    # IterableDatasets, and as such are infinite.
    val_check_interval: Union[int, float] = 100


@dataclass
class RLMethod(Method, target_setting=RLSetting):
    """ Method aimed at solving an RL setting. """
    
    # Options for the Trainer object.
    trainer_options: RLTrainerConfig = mutable_field(RLTrainerConfig)

    def model_class(self, setting: RLSetting) -> Type[Agent]:
        """Retuns the type of LightningModule to use for a given setting.

        In this case, we return the 'Agent', but you can overwrite this in your
        subclass if you want to use a different module.

        Args:
            setting (RLSetting): A setting to be evaluated on.

        Returns:
            Type[Agent]: The type of LightningModule to use for that setting.
        """
        return ActorCritic


if __name__ == "__main__":
    RLMethod.main()
