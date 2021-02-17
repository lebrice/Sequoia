from sequoia.methods import Method
from sequoia.settings import (
    ClassIncrementalSetting,
    DomainIncrementalSetting,
    Setting,
    TaskIncrementalSetting,
)

from .iid_setting import IIDSetting


class MethodA(Method, target_setting=ClassIncrementalSetting):
    pass


class MethodB(Method, target_setting=DomainIncrementalSetting):
    pass


class MethodC(Method, target_setting=TaskIncrementalSetting):
    pass


class MethodD(Method, target_setting=IIDSetting):
    pass


def test_methods_applicable_to_iid_setting():
    """ Test to make sure that Methods that are applicable to the Domain-Incremental
    are applicable to the IID Setting, same for those targetting the Task-Incremental
    setting.
    """
    assert MethodA.is_applicable(ClassIncrementalSetting)
    assert MethodA.is_applicable(TaskIncrementalSetting)
    assert MethodA.is_applicable(DomainIncrementalSetting)
    assert MethodA.is_applicable(IIDSetting)

    assert not MethodB.is_applicable(ClassIncrementalSetting)
    assert not MethodB.is_applicable(TaskIncrementalSetting)
    assert MethodB.is_applicable(DomainIncrementalSetting)
    assert MethodB.is_applicable(IIDSetting)

    assert not MethodC.is_applicable(ClassIncrementalSetting)
    assert MethodC.is_applicable(TaskIncrementalSetting)
    assert not MethodC.is_applicable(DomainIncrementalSetting)
    assert MethodC.is_applicable(IIDSetting)

    assert not MethodD.is_applicable(ClassIncrementalSetting)
    assert not MethodD.is_applicable(TaskIncrementalSetting)
    assert not MethodD.is_applicable(DomainIncrementalSetting)
    assert MethodD.is_applicable(IIDSetting)


def test_get_parents():
    assert IIDSetting in TaskIncrementalSetting.get_children()
    assert IIDSetting in DomainIncrementalSetting.get_children()
    assert IIDSetting not in ClassIncrementalSetting.get_children()
    
    assert TaskIncrementalSetting in IIDSetting.get_immediate_parents()
    assert DomainIncrementalSetting in IIDSetting.get_immediate_parents()
    assert ClassIncrementalSetting not in IIDSetting.get_immediate_parents()
    
    assert TaskIncrementalSetting in IIDSetting.get_parents()
    assert DomainIncrementalSetting in IIDSetting.get_parents()
    assert ClassIncrementalSetting in IIDSetting.get_parents()
    assert IIDSetting not in IIDSetting.get_parents()
