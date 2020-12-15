from dataclasses import dataclass

import pytest

from sequoia.settings import Setting

from sequoia.settings.base import Method

@dataclass
class NewSetting(Setting):
    pass

@dataclass
class NewMethod(Method, target_setting=NewSetting):
    def fit(self, train_env, valid_env):
        pass
    def get_actions(self, observations, action_space):
        return action_space.sample()
    pass


def test_passing_arg_to_class_constructor_works():
    assert NewMethod.target_setting is NewSetting
    assert NewMethod().target_setting is NewSetting


@pytest.mark.xfail(reason="Not sure this is necessary.")
def test_cant_change_target_setting():
    with pytest.raises(AttributeError):
        NewMethod.target_setting = NewSetting
    with pytest.raises(AttributeError):
        NewMethod().target_setting = NewSetting


def test_target_setting_is_inherited():
    @dataclass
    class NewMethod2(NewMethod):
        pass
    assert NewMethod2.target_setting is NewSetting


@dataclass
class SettingA(Setting): pass

@dataclass
class SettingA1(SettingA): pass

@dataclass
class SettingA2(SettingA): pass

@dataclass
class SettingB(Setting): pass


class MethodA(Method, target_setting=SettingA):
    def fit(self, train_env, valid_env):
        pass
    def get_actions(self, observations, action_space):
        return action_space.sample()

class MethodB(Method, target_setting=SettingB):
    def fit(self, train_env, valid_env):
        pass
    def get_actions(self, observations, action_space):
        return action_space.sample()

class CoolGeneralMethod(Method, target_setting=Setting):
    def fit(self, train_env, valid_env):
        pass
    def get_actions(self, observations, action_space):
        return action_space.sample()

def test_method_is_applicable_to_setting():
    """Test the mechanism for determining if a method is applicable for a given
    setting.

    Uses the mock hierarchy created above:
    - Setting
        - SettingA
            - SettingA1
            - SettingA2
        - SettingB

    - Method
        - MethodA (target_setting: SettingA)
        - MethodB (target_setting: SettingA)

    TODO: if we ever end up registering the method classes when declaring them,
    then we will need to check that this dummy test hierarchy doesn't actually
    show up in the real setting options.
    """
    # A Method designed for `SettingA` ISN'T applicable on the root node
    # `Setting`: 
    assert not MethodA.is_applicable(Setting)
    
    # A Method designed for `SettingA` IS applicable on the target node, and all
    # nodes below it in the tree: 
    assert MethodA.is_applicable(SettingA)
    assert MethodA.is_applicable(SettingA1)
    assert MethodA.is_applicable(SettingA2)
    # A Method designed for `SettingA` ISN'T applicable on some other branch in
    # the tree: 
    assert not MethodA.is_applicable(SettingB)

    # Same for Method designed for `SettingB`
    assert MethodB.is_applicable(SettingB)
    assert not MethodB.is_applicable(Setting)
    assert not MethodB.is_applicable(SettingA)
    assert not MethodB.is_applicable(SettingA1)
    assert not MethodB.is_applicable(SettingA2)


def test_is_applicable_also_works_on_instances():
    assert MethodA().is_applicable(SettingA)
    assert MethodA.is_applicable(SettingA())
    assert MethodA().is_applicable(SettingA())

    assert not MethodA().is_applicable(SettingB)
    assert not MethodA.is_applicable(SettingB())
    assert not MethodA().is_applicable(SettingB())