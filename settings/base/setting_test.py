
from dataclasses import dataclass
from .setting import Setting
from utils import constant

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


import pytest


def test_settings_override_with_constant_take_init():
    bob1 = Setting1(foo=3, bar=7)
    assert bob1.foo == 3
    assert bob1.bar == 7
    bob2 = Setting2(foo=4, bar=4)
    assert bob2.bar == 1.0
    assert bob2.foo == 4

def test_init_still_works():
    setting = Setting(val_fraction=0.01)
    assert setting.val_fraction == 0.01
