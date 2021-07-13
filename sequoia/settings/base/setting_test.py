
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


@pytest.mark.xfail(reason="Changed this.")
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


def test_loading_benchmark_doesnt_overwrite_constant():
    setting1 = Setting1.loads_json('{"foo":1, "bar":2}')
    assert setting1.foo == 1
    assert setting1.bar == 2

    setting2 = Setting2.loads_json('{"foo":1, "bar":2}')
    assert setting2.foo == 1
    assert setting2.bar == 1


def test_init_still_works():
    setting = Setting(val_fraction=0.01)
    assert setting.val_fraction == 0.01


def test_passing_unexpected_arg_raises_typeerror():
    with pytest.raises(TypeError):
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


from typing import ClassVar, Type, Dict, Any
from sequoia.common.config import Config
from sequoia.methods.random_baseline import RandomBaselineMethod
from sequoia.settings.base.results import Results
from .setting import Setting


class SettingTests:
    """ Tests for a Setting. """
    Setting: ClassVar[Type[Setting]]

    # TODO: How to parametrize this dynamically based on the value of `Setting`?
    
    # The kwargs to be passed to the Setting when we want to create a 'short' setting.
    fast_dev_run_kwargs: ClassVar[Dict[str, Any]]

    def assert_chance_level(self, setting: Setting, results: Setting.Results):
        """Called during testing. Use this to assert that the results you get
        from applying your method on the given setting match your expectations.

        Args:
            setting
            results (Results): A given Results object.
        """
        assert results is not None
        assert results.objective > 0
        print(
            f"Objective when applied to a setting of type {type(setting)}: {results.objective}"
        )

    @pytest.mark.timeout(60)
    def test_random_baseline(self, config: Config):
        """
        Test that applies a random baseline to the Setting, and checks that the results
        are around chance level.
        """
        # Create the Setting
        setting_type = self.Setting
        # if issubclass(setting_type, ContinualRLSetting):
        #     kwargs.update(max_steps=100, test_steps_per_task=100)
        # if issubclass(setting_type, IncrementalRLSetting):
        #     kwargs.update(nb_tasks=2)
        # if issubclass(setting_type, ClassIncrementalSetting):
        #     kwargs = dict(nb_tasks=5)
        # if issubclass(setting_type, (TraditionalSLSetting, RLSetting)):
        #     kwargs.pop("nb_tasks", None)
        # if isinstance(setting, SLSetting):
        #     method.batch_size = 64
        # elif isinstance(setting, RLSetting):
        #     method.batch_size = None
        #     setting.max_steps = 100

        setting: Setting = setting_type(**self.fast_dev_run_kwargs)
        method = RandomBaselineMethod()

        results = setting.apply(method, config=config)
        self.assert_chance_level(setting, results=results)
