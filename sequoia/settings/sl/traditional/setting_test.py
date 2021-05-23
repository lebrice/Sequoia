from sequoia.methods import Method
from sequoia.settings import (
    ClassIncrementalSetting,
    DomainIncrementalSLSetting,
    Setting,
    TaskIncrementalSLSetting,
)
import pytest

from .setting import TraditionalSLSetting


class MethodA(Method, target_setting=ClassIncrementalSetting):
    pass


class MethodB(Method, target_setting=DomainIncrementalSLSetting):
    pass


class MethodC(Method, target_setting=TaskIncrementalSLSetting):
    pass


class MethodD(Method, target_setting=TraditionalSLSetting):
    pass


def test_methods_applicable_to_iid_setting():
    """ Test to make sure that Methods that are applicable to the Domain-Incremental
    are applicable to the IID Setting, same for those targetting the Task-Incremental
    setting.
    """
    assert MethodA.is_applicable(ClassIncrementalSetting)
    assert MethodA.is_applicable(TaskIncrementalSLSetting)
    assert MethodA.is_applicable(DomainIncrementalSLSetting)
    assert MethodA.is_applicable(TraditionalSLSetting)

    assert not MethodC.is_applicable(ClassIncrementalSetting)
    assert MethodC.is_applicable(TaskIncrementalSLSetting)
    assert not MethodC.is_applicable(DomainIncrementalSLSetting)
    assert MethodC.is_applicable(TraditionalSLSetting)

    assert not MethodD.is_applicable(ClassIncrementalSetting)
    assert not MethodD.is_applicable(TaskIncrementalSLSetting)
    assert not MethodD.is_applicable(DomainIncrementalSLSetting)
    assert MethodD.is_applicable(TraditionalSLSetting)


def test_get_parents():
    assert TraditionalSLSetting in TaskIncrementalSLSetting.get_children()
    assert TraditionalSLSetting not in ClassIncrementalSetting.immediate_children()

    assert TaskIncrementalSLSetting in TraditionalSLSetting.parents()
    assert ClassIncrementalSetting in TaskIncrementalSLSetting.immediate_parents()

    assert TaskIncrementalSLSetting in TraditionalSLSetting.get_parents()
    assert ClassIncrementalSetting in TraditionalSLSetting.get_parents()
    assert TraditionalSLSetting not in TraditionalSLSetting.get_parents()


@pytest.mark.xfail(
    reason="Temporarily removing the domain-incremental<--traditional link."
)
def test_get_parents_domain_incremental():
    assert TraditionalSLSetting in DomainIncrementalSLSetting.get_children()
    assert DomainIncrementalSLSetting in TraditionalSLSetting.get_immediate_parents()


@pytest.mark.xfail(
    reason="Temporarily removing the domain-incremental<--traditional link."
)
def test_method_applicability_domain_incremental():
    assert not MethodB.is_applicable(ClassIncrementalSetting)
    assert not MethodB.is_applicable(TaskIncrementalSLSetting)
    assert MethodB.is_applicable(DomainIncrementalSLSetting)
    assert MethodB.is_applicable(TraditionalSLSetting)


@pytest.mark.xfail(
    reason="Temporarily removing the domain-incremental<--traditional link."
)
def test_get_parents_domain_incremental():
    assert DomainIncrementalSLSetting in TraditionalSLSetting.get_parents()
