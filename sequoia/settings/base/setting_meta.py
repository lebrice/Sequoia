"""

"""
import dataclasses
import inspect
import sys
import traceback
from dataclasses import Field
from typing import Type, List, Dict, Any
from pytorch_lightning.core.datamodule import _DataModuleWrapper
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)

class SettingMeta(_DataModuleWrapper, Type["Setting"]):
    """ Metaclass for the nodes in the Setting inheritance tree.
    
    Might remove this. Was experimenting with using this to create class
    properties for each Setting.
    
    What this currently does is to remove any keyword argument passed to the
    constructor if its value is marked as a 'constant'.

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
        fields: Dict[str, Field] = {
            field.name: field for field in dataclasses.fields(cls)
        }
        init_fields: List[str] = [name for name, f in fields.items() if f.init]
        
        for key in list(kwargs.keys()):
            value = kwargs[key]
            if key not in fields:
                # We let this through, so that if there is a problem, it is
                # raised when calling the constructor below.
                continue
            # elif key in fields and key not in init_fields:
            #     # We let this through, so that if there is a problem, it is
            #     # raised when calling the constructor below.
            #     logger.warning(RuntimeWarning(
            #         f"Constructor Argument {key} is a field with init=False but"
            #         f"but is being passed to the constructor."
            #     ))
            #     continue
                # Alternative: Raise a custom Exception directly:
                # raise RuntimeError((
                # Other idea: go up two stackframes so that it looks like
                # `cls(blabla=123)` is what's causing the exception?

            field = fields[key]
            _missing = object()
            constant_value = field.metadata.get("constant", _missing)
            if constant_value is not _missing and value != constant_value:
                logger.warning(UserWarning(
                    f"Ignoring argument {key}={value} when creating class "
                    f"{cls}, since it has that field marked as constant with a "
                    f"value of {constant_value}."
                ))
                kwargs.pop(key)
        return super().__call__(*args, **kwargs)
