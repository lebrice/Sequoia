from abc import ABC, abstractmethod
from typing import Optional, Union, List, Type, ClassVar, Any

import gym
import numpy as np
from pytorch_lightning import LightningDataModule

from .base import Actions, Environment, Observations, Rewards


class MethodABC(ABC):
    """ ABC for a Method, which is a solution to a research problem (a Setting).
    """
    # Class attribute that holds the setting this method was designed to target.
    # Needs to either be passed to the class statement or set as a class
    # attribute.
    target_setting: ClassVar[Type["SettingABC"]]

    @abstractmethod
    def get_actions(self, observations: Observations, action_space: gym.Space) -> Union[Actions, Any]:
        """ Get a batch of predictions (actions) for the given observations.
        returned actions must fit the action space.
        """

    @abstractmethod
    def fit(self,
            train_dataloader: Environment[Observations, Actions, Rewards] = None,
            valid_dataloader: Environment[Observations, Actions, Rewards] = None,
            datamodule: LightningDataModule = None):
        """Called by the Setting to train the method.

        Might be called more than once before training is 'done'.
        """

    ## Below this are some class attributes and methods related to the Tree.

    @classmethod
    def is_applicable(cls, setting: Union["SettingABC", Type["SettingABC"]]) -> bool:
        """Returns wether this Method is applicable to the given setting.

        A method is applicable on a given setting if and only if the setting is
        the method's target setting, or if it is a descendant of the method's
        target setting (below the target setting in the tree).
        
        Concretely, since the tree is implemented as an inheritance hierarchy,
        a method is applicable to any setting which is an instance (or subclass)
        of its target setting.

        Args:
            setting (SettingType): a Setting.

        Returns:
            bool: Wether or not this method is applicable on the given setting.
        """
        from .setting_abc import SettingABC
        
        # if given an object, get it's type.
        if isinstance(setting, LightningDataModule):
            setting = type(setting)
        
        if (not issubclass(setting, SettingABC)
            and issubclass(setting, LightningDataModule)):
            # TODO: If we're trying to check if this method would be compatible
            # with a LightningDataModule, rather than a Setting, then we treat
            # that LightningModule the same way we would an IIDSetting.
            # i.e., if we're trying to apply a Method on something that isn't in
            # the tree, then we consider that datamodule as the IIDSetting node.
            from settings import IIDSetting
            setting = IIDSetting

        return issubclass(setting, cls.target_setting)

    @classmethod
    def get_applicable_settings(cls) -> List[Type["SettingABC"]]:
        """ Returns all settings on which this method is applicable.
        NOTE: This only returns 'concrete' Settings.
        """
        from settings import all_settings
        return list(filter(cls.is_applicable, all_settings))
        # This would return ALL the setting:
        # return list([cls.target_setting, *cls.target_setting.all_children()])

    @classmethod
    def get_name(cls) -> str:
        """ Gets the name of this method class. """
        name = getattr(cls, "name", None)
        if name is None:
            from utils import camel_case, remove_suffix
            name = camel_case(cls.__qualname__)
            name = remove_suffix(name, "_method")
        return name

    def __init_subclass__(cls, target_setting: Type["SettingABC"]=None):
        """Called when creating a new subclass of Method.

        Args:
            target_setting (Type[Setting], optional): The target setting.
                Defaults to None, in which case the method will inherit the
                target setting of it's parent class.
        """
        cls.target_setting = target_setting
        return super().__init_subclass__()
