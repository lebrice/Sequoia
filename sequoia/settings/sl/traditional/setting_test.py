from sequoia.methods import Method
from sequoia.settings import (
    ClassIncrementalSetting,
    DomainIncrementalSLSetting,
    Setting,
    TaskIncrementalSLSetting,
)
import pytest

from .setting import TraditionalSLSetting
from ..multi_task.setting import MultiTaskSLSetting
from ..discrete.setting import DiscreteTaskAgnosticSLSetting
from ..continual.setting import ContinualSLSetting
from ..incremental.setting import IncrementalSLSetting


class ContinualSLMethod(Method, target_setting=ContinualSLSetting):
    pass


class DiscreteTaskAgnosticSLMethod(
    Method, target_setting=DiscreteTaskAgnosticSLSetting
):
    pass


class IncrementalSLMethod(Method, target_setting=IncrementalSLSetting):
    pass


class ClassIncrementalSLMethod(Method, target_setting=ClassIncrementalSetting):
    pass


class DomainIncrementalSLMethod(Method, target_setting=DomainIncrementalSLSetting):
    pass


class TaskIncrementalSLMethod(Method, target_setting=TaskIncrementalSLSetting):
    pass


class TraditionalSLMethod(Method, target_setting=TraditionalSLSetting):
    pass


class MultiTaskSLMethod(Method, target_setting=MultiTaskSLSetting):
    pass


def test_methods_applicable_to_iid_setting():
    """ Test to make sure that Methods that are applicable to the Domain-Incremental
    are applicable to the IID Setting, same for those targetting the Task-Incremental
    setting.
    """
    assert ContinualSLMethod.is_applicable(ContinualSLSetting)
    assert ContinualSLMethod.is_applicable(DiscreteTaskAgnosticSLSetting)
    assert ContinualSLMethod.is_applicable(IncrementalSLSetting)
    assert ContinualSLMethod.is_applicable(ClassIncrementalSetting)
    assert ContinualSLMethod.is_applicable(TaskIncrementalSLSetting)
    assert ContinualSLMethod.is_applicable(DomainIncrementalSLSetting)
    assert ContinualSLMethod.is_applicable(TraditionalSLSetting)
    assert ContinualSLMethod.is_applicable(MultiTaskSLSetting)

    assert not DiscreteTaskAgnosticSLMethod.is_applicable(ContinualSLSetting)
    assert DiscreteTaskAgnosticSLMethod.is_applicable(DiscreteTaskAgnosticSLSetting)
    assert DiscreteTaskAgnosticSLMethod.is_applicable(IncrementalSLSetting)
    assert DiscreteTaskAgnosticSLMethod.is_applicable(ClassIncrementalSetting)
    assert DiscreteTaskAgnosticSLMethod.is_applicable(TaskIncrementalSLSetting)
    assert DiscreteTaskAgnosticSLMethod.is_applicable(DomainIncrementalSLSetting)
    assert DiscreteTaskAgnosticSLMethod.is_applicable(TraditionalSLSetting)
    assert DiscreteTaskAgnosticSLMethod.is_applicable(MultiTaskSLSetting)

    assert not IncrementalSLMethod.is_applicable(ContinualSLSetting)
    assert not IncrementalSLMethod.is_applicable(DiscreteTaskAgnosticSLSetting)
    assert IncrementalSLMethod.is_applicable(IncrementalSLSetting)
    assert IncrementalSLMethod.is_applicable(ClassIncrementalSetting)
    assert IncrementalSLMethod.is_applicable(TaskIncrementalSLSetting)
    assert IncrementalSLMethod.is_applicable(DomainIncrementalSLSetting)
    assert IncrementalSLMethod.is_applicable(TraditionalSLSetting)
    assert IncrementalSLMethod.is_applicable(MultiTaskSLSetting)

    assert not ClassIncrementalSLMethod.is_applicable(ContinualSLSetting)
    assert not ClassIncrementalSLMethod.is_applicable(DiscreteTaskAgnosticSLSetting)
    assert ClassIncrementalSLMethod.is_applicable(IncrementalSLSetting)
    assert ClassIncrementalSLMethod.is_applicable(ClassIncrementalSetting)
    assert ClassIncrementalSLMethod.is_applicable(TaskIncrementalSLSetting)
    assert ClassIncrementalSLMethod.is_applicable(DomainIncrementalSLSetting)
    assert ClassIncrementalSLMethod.is_applicable(TraditionalSLSetting)
    assert ClassIncrementalSLMethod.is_applicable(MultiTaskSLSetting)

    assert not TaskIncrementalSLMethod.is_applicable(ContinualSLSetting)
    assert not TaskIncrementalSLMethod.is_applicable(DiscreteTaskAgnosticSLSetting)
    assert not TaskIncrementalSLMethod.is_applicable(IncrementalSLSetting)
    assert not TaskIncrementalSLMethod.is_applicable(ClassIncrementalSetting)
    assert TaskIncrementalSLMethod.is_applicable(TaskIncrementalSLSetting)
    assert not TaskIncrementalSLMethod.is_applicable(DomainIncrementalSLSetting)
    assert not TaskIncrementalSLMethod.is_applicable(TraditionalSLSetting)
    assert TaskIncrementalSLMethod.is_applicable(MultiTaskSLSetting)

    assert not DomainIncrementalSLMethod.is_applicable(ContinualSLSetting)
    assert not DomainIncrementalSLMethod.is_applicable(DiscreteTaskAgnosticSLSetting)
    assert not DomainIncrementalSLMethod.is_applicable(IncrementalSLSetting)
    assert not DomainIncrementalSLMethod.is_applicable(ClassIncrementalSetting)
    assert not DomainIncrementalSLMethod.is_applicable(TaskIncrementalSLSetting)
    assert DomainIncrementalSLMethod.is_applicable(DomainIncrementalSLSetting)
    assert not DomainIncrementalSLMethod.is_applicable(TraditionalSLSetting)
    # TODO: What about this one?
    # assert DomainIncrementalSLMethod.is_applicable(MultiTaskSLSetting)

    assert not TraditionalSLMethod.is_applicable(ContinualSLSetting)
    assert not TraditionalSLMethod.is_applicable(DiscreteTaskAgnosticSLSetting)
    assert not TraditionalSLMethod.is_applicable(IncrementalSLSetting)
    assert not TraditionalSLMethod.is_applicable(TaskIncrementalSLSetting)
    assert not TraditionalSLMethod.is_applicable(DomainIncrementalSLSetting)
    assert not TraditionalSLMethod.is_applicable(ClassIncrementalSetting)
    assert TraditionalSLMethod.is_applicable(TraditionalSLSetting)
    assert TraditionalSLMethod.is_applicable(MultiTaskSLSetting)

    assert not MultiTaskSLMethod.is_applicable(ContinualSLSetting)
    assert not MultiTaskSLMethod.is_applicable(DiscreteTaskAgnosticSLSetting)
    assert not MultiTaskSLMethod.is_applicable(IncrementalSLSetting)
    assert not MultiTaskSLMethod.is_applicable(TaskIncrementalSLSetting)
    assert not MultiTaskSLMethod.is_applicable(DomainIncrementalSLSetting)
    assert not MultiTaskSLMethod.is_applicable(ClassIncrementalSetting)
    assert not MultiTaskSLMethod.is_applicable(TraditionalSLSetting)
    assert MultiTaskSLMethod.is_applicable(MultiTaskSLSetting)


def test_get_parents():
    # TODO: THis is a bit funky, now that Class-Incremental is a "pointer" to
    # Incremental, and Traditional has been moved under TaskIncremental
    assert TraditionalSLSetting in IncrementalSLSetting.get_children()
    assert TraditionalSLSetting not in TaskIncrementalSLSetting.get_children()
    assert TraditionalSLSetting in IncrementalSLSetting.immediate_children()

    assert TaskIncrementalSLSetting not in TraditionalSLSetting.parents()
    assert ClassIncrementalSetting in TaskIncrementalSLSetting.immediate_parents()

    assert TaskIncrementalSLSetting not in TraditionalSLSetting.get_parents()
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
    assert not DomainIncrementalSLMethod.is_applicable(ClassIncrementalSetting)
    assert not DomainIncrementalSLMethod.is_applicable(TaskIncrementalSLSetting)
    assert DomainIncrementalSLMethod.is_applicable(DomainIncrementalSLSetting)
    assert DomainIncrementalSLMethod.is_applicable(TraditionalSLSetting)


@pytest.mark.xfail(
    reason="Temporarily removing the domain-incremental<--traditional link."
)
def test_get_parents_domain_incremental():
    assert DomainIncrementalSLSetting in TraditionalSLSetting.get_parents()
