import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sequoia.settings.sl import TaskIncrementalSLSetting
from sequoia.settings.sl.continual import Observations, Rewards, Environment
from sequoia.methods.packnet_method import PackNetMethod
from typing import Tuple, Optional


class SmallerSequoiaClassifier(pl.LightningModule):

    def __init__(self, input_channels=1):
        super(SmallerSequoiaClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=5)
        self.norm_layer = nn.BatchNorm2d(num_features=3, affine=True)
        self.dense1 = nn.Linear(in_features=1728, out_features=10)

    def forward(self, x):
        """
        :param x: 1x28x28 tensor representing MNIST image
        :return: logits, 10 classes
        """
        x = F.relu(self.norm_layer(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batchidx):
        observations, rewards = batch
        x = observations.x

        logits = self(x)
        y_pred = logits.argmax(-1)
        if rewards is None:
            # NOTE: See the pl_example.py in `sequoia/examples/basic/pl_example.py` for
            # more info about when this might happen.
            environment: Environment = self.trainer.request_dataloader("train")
            rewards = environment.send(y_pred)

        assert rewards is not None
        y = rewards.y
        accuracy = (y_pred == y).int().sum().div(len(y))
        self.log("train/accuracy", accuracy, prog_bar=True)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


if __name__ == '__main__':

    setting = TaskIncrementalSLSetting(
        dataset="mnist"
    )

    m = SmallerSequoiaClassifier(input_channels=3)

    my_method = PackNetMethod(model=m, prune_instructions=.7, epoch_split=(3, 1))
    results = setting.apply(my_method)
