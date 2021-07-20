from .model import Model
from gym.spaces import Box
import torch.nn as nn


class Baseline(Model):
    def __init__(self, image_space: Box, n_classes: int) -> None:
        # assertion for Conv networks
        assert len(image_space.shape) == 3
        modules_list, penulimate_layer_indx = self._get_modules(image_space, n_classes)
        super(Baseline, self).__init__(modules_list, penulimate_layer_indx)

    def _get_modules(self, image_space: Box, n_classes: int):
        image_channels = image_space.shape[0]
        encoder = []
        encoder.append(self.get_encoder_layer(image_channels, 6, 5))

        encoder.append(self.get_encoder_layer(6, 16, 5))
        rand_tensor = self.get_random_input(image_space.shape)
        representation_size = self._compute_linear_input(encoder, rand_tensor)
        classifier = []
        classifier.append(
            nn.Sequential(
                *[nn.Flatten(), nn.Linear(representation_size, 120), nn.ReLU(True)]
            )
        )
        classifier.append(nn.Sequential(*[nn.Linear(120, 84), nn.ReLU(True)]))
        classifier.append(nn.Linear(84, n_classes))
        moddules_list = encoder + classifier
        penulimate_layer_indx = len(encoder)
        return moddules_list, penulimate_layer_indx

    def get_encoder_layer(self, in_channels, out_channels, kernel_size):
        layer = []
        layer.append(nn.Conv2d(in_channels, out_channels, kernel_size))
        layer.append(nn.ReLU(True))
        layer.append(nn.MaxPool2d(2))
        return nn.Sequential(*layer)
