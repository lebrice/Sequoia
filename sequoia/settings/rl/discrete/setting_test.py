from ..continual.setting_test import TestContinualRLSetting as ContinualRLSettingTests
from .setting import DiscreteTaskAgnosticRLSetting
from typing import ClassVar, Type 
from sequoia.settings import Setting


class TestDiscreteTaskAgnosticRLSetting(ContinualRLSettingTests):
    Setting: ClassVar[Type[Setting]] = DiscreteTaskAgnosticRLSetting

    # IDEA: Create a fixture that creates the Setting which can then be tested.
    # TODO: Maybe this is a bit too complicated..