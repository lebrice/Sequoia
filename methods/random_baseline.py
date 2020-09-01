"""A random baseline Method that gives random predictions for any input.

Should be applicable to any Setting.
"""
from abc import ABC
from dataclasses import dataclass
from typing import Type, Union

import torch
from singledispatchmethod import singledispatchmethod
from torch import Tensor

from methods.method import Method
from methods.models import Model, OutputHead
from methods.models.class_incremental_model import ClassIncrementalModel
from methods.models.iid_model import IIDModel
from methods.models.task_incremental_model import TaskIncrementalModel
from settings import (ClassIncrementalSetting, IIDSetting, Setting,
                      SettingType, TaskIncrementalSetting)
from utils import get_logger

from .models import HParams, Model

logger = get_logger(__file__)


class RandomOutputHead(OutputHead):
    def forward(self, h_x: Tensor):
        batch_size = h_x.shape[0]
        return torch.rand([batch_size, self.output_size], requires_grad=True).type_as(h_x)

class RandomPredictionsMixin(ABC):
    """ A mixin class that when applied to a Model class makes it give random
    predictions.
    """
    def encode(self, x: Tensor):
        """ Gives back a random encoding instead of doing a forward pass through
        the encoder.
        """
        batch_size = x.shape[0]
        h_x = torch.rand([batch_size, self.hidden_size])
        return h_x.type_as(x)

    @property
    def output_head_class(self) -> Type[OutputHead]:
        """Property which returns the type of output head to use.

        overwrite this if your model does something different than classification.

        Returns:
            Type[OutputHead]: A subclass of OutputHead.
        """
        return RandomOutputHead


class RandomClassIncrementalModel(RandomPredictionsMixin, ClassIncrementalModel):
    pass


class RandomTaskIncrementalModel(RandomPredictionsMixin, TaskIncrementalModel):
    pass


class RandomIIDModel(RandomPredictionsMixin, IIDModel):
    pass


def get_random_model_class(base_model_class: Type[Model]) -> Type[Union[RandomPredictionsMixin, Model]]:
    class RandomModel(RandomPredictionsMixin, base_model_class):
        pass
    return RandomModel

@dataclass
class RandomBaselineMethod(Method, target_setting=Setting):
    """ Baseline method that gives random predictions for any given setting.

    We do this by creating a base Model with an output head that gives random
    predictions.
    
    TODO: Actually make this compatible with other settings than
    task-incremental and iid. There will probably be some code shuffling to do
    with respect to the `Model` class, as it is moreso aimed at being a `passive`
    Model than an active one at the moment.
    """
    @singledispatchmethod
    def model_class(self, setting: SettingType) -> Type[Model]:
        raise NotImplementedError(f"No known model for setting {setting}!")

    @model_class.register
    def _(self, setting: ClassIncrementalSetting) -> Type[ClassIncrementalModel]:
        # IDEA: Generate the model dynamically instead of creating one of each.
        # (This doesn't work atm because super() gives back a Model)
        # return get_random_model_class(super().model_class(setting))
        return RandomClassIncrementalModel

    @model_class.register
    def _(self, setting: TaskIncrementalSetting) -> Type[TaskIncrementalModel]:
        return RandomTaskIncrementalModel

    @model_class.register
    def _(self, setting: IIDSetting) -> Type[IIDModel]:
        return RandomIIDModel

    def create_model(self, setting: SettingType) -> Model[SettingType]:
        """ Create the baseline model. """
        # Get the type of model to use for that setting.
        model_class: Type[Model] = self.model_class(setting)
        hparams_class = model_class.HParams
        logger.debug(f"model class for this setting: {model_class}")
        logger.debug(f"hparam class for this setting: {hparams_class}")
        logger.debug(f"Hyperparameters class on the method: {type(self.hparams)}")

        if not isinstance(self.hparams, hparams_class):
            # TODO: @lebrice This is ugly, and should be cleaned up somehow. Let
            # me know what you think:
            #
            # The problem is that in order to have the --help option display all
            # the options for the Method (including the model hparams), the
            # hparams should be one or more fields on the Method object.
            #
            # However, if in our method we use a different Model class depending
            # on the type of Setting, then we would need the hyperparameters to
            # be of the type required by the model!
            #
            # Therefore, here we upgrade `self.hparams` (if present) to the
            # right type (`model_class.HParams`)
            logger.warning(UserWarning(
                f"The hparams attribute on the {self.get_name()} Method are of "
                f"type {type(self.hparams)}, while the HParams on the model "
                f"class are of type {hparams_class}!\n"
                f"This will try to 'upgrade' the hparams, using values "
                f"from the command-line."
            ))
            self.hparams = self.upgrade_hparams(hparams_class)
            logger.info(f"'Upgraded' hparams: {self.hparams}")

        assert isinstance(self.hparams, model_class.HParams)
        return model_class(setting=setting, hparams=self.hparams, config=self.config)

    def upgrade_hparams(self, new_type: Type[HParams]) -> HParams:
        """Upgrades the current hparams to the new type, filling in the new
        values from the command-line.

        Args:
            new_type (Type[HParams]): Type of HParams to upgrade to.
            argv (Union[str, List[str]], optional): Command-line arguments to
            use to set the missing values. Defaults to None, in which case the
            values in `sys.argv` are used.

        Returns:
            HParams: [description]
        """
        argv = self._argv
        logger.info(f"Current method was originally created from args {argv}")
        new_hparams: HParams = new_type.from_args(argv)
        logger.info(f"Hparams for that type of model (from the method): {self.hparams}")
        logger.info(f"Hparams for that type of model (from command-line): {new_hparams}")
        
        # if self.hparams:
        #     # IDEA: use some fancy dict comparisons to keep things that aren't the same
        #     # Not needed, because we saved the args that were used to create the instance.
        #     default_values = self.hparams.from_dict({})
        #     current_values = self.hparams.to_dict()
        #     different_values = utils.
        #     new_hparams = new_type.from_dict(hparams_dict, drop_extra_fields=True)
        return new_hparams


if __name__ == "__main__":
    RandomBaselineMethod.main()
