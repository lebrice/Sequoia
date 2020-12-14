from .active_setting import ActiveSetting
from .active_dataloader import ActiveDataLoader
ActiveEnvironment = ActiveDataLoader
from .rl import (ClassIncrementalRLSetting, ContinualRLSetting, RLSetting,
                 TaskIncrementalRLSetting)

