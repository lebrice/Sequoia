from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch.nn import functional as F
from torch import Tensor, nn


class GenerativeModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()