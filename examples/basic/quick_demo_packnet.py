import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sequoia.settings.sl import TaskIncrementalSLSetting, ContinualSLSetting
from sequoia.settings.sl.continual import Observations, Rewards, Environment
from sequoia.methods.packnet_method import PackNetMethod
from examples.basic.pl_example import Model
from typing import Tuple, Optional

if __name__ == '__main__':
    setting = TaskIncrementalSLSetting(
        dataset="mnist"
    )

    my_method = PackNetMethod()

    results = setting.apply(my_method)
