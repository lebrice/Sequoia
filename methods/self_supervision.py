from dataclasses import dataclass
from functools import lru_cache
# class SelfSupervisedAgent(SelfSupervisedModel, Agent):
#     pass
from typing import ClassVar, Dict, List, Optional, Type, Union

from common.config import Config
from common.loss import Loss
from common.tasks import (AEReconstructionTask, AuxiliaryTask, EWCTask,
                          SimCLRTask, VAEReconstructionTask)
from common.tasks.simclr import SimCLRTask
from pytorch_lightning import Callback
from settings import (ClassIncrementalSetting, IIDSetting, RLSetting, Setting,
                      SettingType, TaskIncrementalSetting)
from simple_parsing import mutable_field
from simple_parsing.helpers import FlattenedAccess
from torch import Tensor
from utils import get_logger, singledispatchmethod
# TODO: Where should the coefficients for the different auxiliary tasks be?
# I think they should be in this file here, but then how will it make sense
from utils.serialization import Serializable

from .class_incremental_method import (ClassIncrementalMethod,
                                       ClassIncrementalModel)
from .method import Method
from .models import Model
from .models.actor_critic_agent import ActorCritic
from .models.agent import Agent
from .models.iid_model import IIDModel
from .models.task_incremental_model import TaskIncrementalModel

logger = get_logger(__file__)


@dataclass
class SelfSupervisionHParams(Serializable):
    """Hyper-parameters of the self-supervised method, including all the tasks.

    TODO: I guess users would add new auxiliary tasks to this list here?
    """
    simclr: Optional[SimCLRTask.Options] = None
    vae: Optional[VAEReconstructionTask.Options] = None
    ae: Optional[AEReconstructionTask.Options] = None
    ewc: Optional[EWCTask.Options] = None


# # TODO: Not sure if this is better than defining all the models above.
# @lru_cache(maxsize=None, typed=True)
# def make_self_supervised_model_class(model_class: Type[Model]) -> Type[Union[SelfSupervisedModel, Model]]:
#     class SelfSupervisedVariant(SelfSupervisedModel, model_class):
#         @dataclass
#         class HParams(SelfSupervisedModel, model_class.HParams):
#             pass
#     SelfSupervisedVariant.__name__ = "SelfSupervised" + model_class.__name__
#     return SelfSupervisedVariant


@dataclass
class SelfSupervision(Method, target_setting=Setting):
    """ Method where self-supervised learning is used to learn representations.

    The representations of the model are learned either jointly with the
    downstream task (e.g. classification) loss, or only through self-supervision
    when `detach_output_head` is set to True.

    TODO: This doesn't really belong here anymore, because the base model is
    self-supervised by default!
    """
    name: ClassVar[str] = "self_supervision"

    # Hyperparameters of the model.
    # TODO: If we were to support more models, we might have a problem trying to
    # get the help text of each type of hyperparameter to show up. We can still
    # parse them just fine by calling .from_args() on them, but still, would be
    # better if the help text were visible from the command-line.
    hparams: Model.HParams = mutable_field(Model.HParams)

    # @singledispatchmethod
    # def model_class(self, setting: Setting):
    #     return NotImplementedError(f"No model registered for setting {setting}")
    
    # @model_class.register
    # def _(self, setting: ClassIncrementalSetting):
    #     return SelfSupervisedClassIncrementalModel
    
    # @model_class.register
    # def _(self, setting: IIDSetting):
    #     return IIDModel

    # @model_class.register
    # def _(self, setting: RLSetting):
    #     return ActorCritic

if __name__ == "__main__":
    SelfSupervision.main()
