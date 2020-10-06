"""

"""
from typing import Type, List, Dict, Any
from pytorch_lightning.core.datamodule import _DataModuleWrapper
from dataclasses import fields
from utils.logging_utils import get_logger

logger = get_logger(__file__)

class SettingMeta(_DataModuleWrapper, Type["Setting"]):
    """ Metaclass for the nodes in the Setting inheritance tree.
    
    Might remove this. Was experimenting with using this to create class
    properties for each Setting.

    TODO: A little while back I noticed some strange behaviour when trying
    to create a Setting class (either manually or through the command-line), and
    I attributed it to PL adding a `_DataModuleWrapper` metaclass to
    `LightningDataModule`, which seemed to be causing problems related to
    calling __init__ when using dataclasses. I don't quite recall exactly what
    was happening and was causing an issue, so it would be a good idea to try
    removing this metaclass and writing a test to make sure there was a problem
    to begin with, and also to make sure that adding back this class fixes it.
    """
    def __call__(cls, *args, **kwargs):
        # This is used to filter the arguments passed to the constructor
        # of the Setting and only keep the ones that are fields with init=True.
        init_fields: List[str] = [f.name for f in fields(cls) if f.init]
        extra_args: Dict[str, Any] = {}
        for k in list(kwargs.keys()):
            if k not in init_fields:
                extra_args[k] = kwargs.pop(k)
        if extra_args:
            logger.warning(UserWarning(
                f"Ignoring args {extra_args} when creating class {cls}."
            ))
        return super().__call__(*args, **kwargs)

