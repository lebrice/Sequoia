from typing import Any, ClassVar, Dict, Type

from sequoia.settings.sl.continual.setting_test import (
    TestContinualSLSetting as ContinualSLSettingTests,
)

from .setting import DiscreteTaskAgnosticSLSetting


class TestDiscreteTaskAgnosticSLSetting(ContinualSLSettingTests):
    Setting: ClassVar[Type[Setting]] = DiscreteTaskAgnosticSLSetting

    # The kwargs to be passed to the Setting when we want to create a 'short' setting.
    fast_dev_run_kwargs: ClassVar[Dict[str, Any]] = dict(
        dataset="mnist",
        batch_size=64,
    )
