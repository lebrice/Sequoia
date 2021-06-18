from .setting import DiscreteTaskAgnosticSLSetting
from typing import ClassVar, Type, Dict, Any
from sequoia.settings.base import Setting
from sequoia.settings.sl.continual.setting_test import TestContinualSLSetting as ContinualSLSettingTests


class TestDiscreteTaskAgnosticSLSetting(ContinualSLSettingTests):
    Setting: ClassVar[Type[Setting]] = DiscreteTaskAgnosticSLSetting

    # The kwargs to be passed to the Setting when we want to create a 'short' setting.
    fast_dev_run_kwargs: ClassVar[Dict[str, Any]] = dict(
        dataset="mnist", batch_size=64,
    )
