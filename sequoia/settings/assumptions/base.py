from sequoia.settings.base.bases import SettingABC
from typing import Type, Any, Union
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)

from pytorch_lightning.core.datamodule import _DataModuleWrapper

# IDEA:  (@lebrice) Exploring the idea of using metaclasses to customize the isinstance
# and subclass checks, so that it could be property-based. This is probably not worth it
# though.
# It's also quite dumb that we have to extend a metaclass from pytorch lightning!

# class AssumptionMeta(_DataModuleWrapper):    
#     def __instancecheck__(self, instance: Union[SettingABC, Any]):
#         logger.debug(f"InstanceCheck on assumption {self} for instance {instance}")
#         return super().__instancecheck__(instance)


class AssumptionBase(SettingABC):
    pass
