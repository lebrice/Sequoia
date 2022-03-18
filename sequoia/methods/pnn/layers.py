import torch.nn as nn
import torch.nn.functional as F

"""
Based on https://github.com/TomVeniat/ProgressiveNeuralNetworks.pytorch
"""


class PNNConvLayer(nn.Module):
    def __init__(self, col, depth, n_in, n_out, kernel_size=3):
        super(PNNConvLayer, self).__init__()
        self.col = col
        self.layer = nn.Conv2d(n_in, n_out, kernel_size, stride=2, padding=1)

        self.u = nn.ModuleList()
        if depth > 0:
            self.u.extend(
                [nn.Conv2d(n_in, n_out, kernel_size, stride=2, padding=1) for _ in range(col)]
            )

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        cur_column_out = self.layer(inputs[-1])
        prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]

        return F.relu(cur_column_out + sum(prev_columns_out))


class PNNLinearBlock(nn.Module):
    def __init__(self, col: int, depth: int, n_in: int, n_out: int):
        super(PNNLinearBlock, self).__init__()
        self.layer = nn.Linear(n_in, n_out)

        self.u = nn.ModuleList()
        if depth > 0:
            self.u.extend([nn.Linear(n_in, n_out) for _ in range(col)])

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        cur_column_out = self.layer(inputs[-1])
        prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]

        return F.relu(cur_column_out + sum(prev_columns_out))
