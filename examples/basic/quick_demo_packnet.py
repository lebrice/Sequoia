import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sequoia.settings.sl import TaskIncrementalSLSetting
from sequoia.settings.sl.continual import Observations, Rewards, Environment
from sequoia.methods.packnet_method import PackNetMethod
from examples.basic.pl_example import Model
from typing import Tuple, Optional

if __name__ == '__main__':
    setting = TaskIncrementalSLSetting(
        dataset="mnist"
    )

    m = Model(input_space=setting.observation_space, output_space=setting.action_space)

    my_method = PackNetMethod(model=m, prune_instructions=.7, epoch_split=(3, 1))

    results = setting.apply(my_method)
