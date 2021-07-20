from gym.spaces.utils import flatdim
from .model import Model
from gym.spaces import Box
import torch.nn as nn


class MLP(Model):
    def __init__(self, image_space: Box, n_classes: int) -> None:
        modules_list, penulimate_layer_indx = self._get_modules(image_space, n_classes)
        super(MLP, self).__init__(modules_list, penulimate_layer_indx)

    def _get_modules(self, image_space: Box, n_classes: int):
        num_inputs = flatdim(image_space)
        hidden_size = 100
        encoder = []
        encoder.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, hidden_size),
             nn.ReLU(inplace=True),
        ))

        decoder = [nn.Linear(hidden_size, n_classes)]
        return encoder + decoder, len(encoder)
