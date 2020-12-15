
from dataclasses import dataclass

import pytest

from sequoia.methods import Method
from sequoia.utils import constant

from .setting import Setting


@dataclass
class Setting1(Setting):
    foo: int = 1
    bar: int = 2

    def __post_init__(self):
        print(f"Setting1 __init__ ({self})")
        super().__post_init__()


@dataclass
class Setting2(Setting1):
    bar: int = constant(1)

    def __post_init__(self):
        print(f"Setting2 __init__ ({self})")
        super().__post_init__()


def test_settings_override_with_constant_take_init():
    """ Test that when a value for one of the constant fields is passed to the
    constructor, its value is ignored and getting that attribute on the object
    gives back the constant value. 
    If the field isn't constant, the value should be set on the object as usual.
    """
    bob1 = Setting1(foo=3, bar=7)
    assert bob1.foo == 3
    assert bob1.bar == 7
    bob2 = Setting2(foo=4, bar=4)
    assert bob2.bar == 1.0
    assert bob2.foo == 4

def test_init_still_works():
    setting = Setting(val_fraction=0.01)
    assert setting.val_fraction == 0.01


def test_passing_unexpected_arg_raises_typeerror():
    with pytest.raises(TypeError, match="unexpected keyword argument 'baz'"):
        bob2 = Setting2(foo=4, bar=4, baz=123123)
        
@dataclass
class SettingA(Setting): pass

@dataclass
class SettingA1(SettingA): pass

@dataclass
class SettingA2(SettingA): pass

@dataclass
class SettingB(Setting): pass

class MethodA(Method, target_setting=SettingA): pass


class MethodB(Method, target_setting=SettingB): pass


class CoolGeneralMethod(Method, target_setting=Setting): pass


def test_that_transforms_can_be_set_through_command_line():
    from sequoia.common.transforms import Transforms, Compose

    setting = Setting(train_transforms=[])
    assert setting.train_transforms == []
    
    
    setting = Setting.from_args("--train_transforms channels_first")
    assert setting.train_transforms == [
        Transforms.channels_first
    ]
    assert isinstance(setting.train_transforms, Compose)
    
    setting = Setting.from_args("--train_transforms channels_first")
    assert setting.train_transforms == [
        Transforms.channels_first
    ]
    assert isinstance(setting.train_transforms, Compose)
