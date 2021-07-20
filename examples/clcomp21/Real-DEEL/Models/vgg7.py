from .model import Model
from gym.spaces import Box
import torch.nn as nn


class VGG7(Model):
    def __init__(self, image_space: Box, n_classes: int):
        # assertion for Conv networks
        assert len(image_space.shape) == 3
        self.use_batchn = True
        self.n_output_classes = n_classes
        encoder_list = []
        n_blocks = [1, 2, 2]
        m_channels_per_block = [128, 256, 512]
        input_channels = image_space.shape[0]
        for channel_indx, n_out_channels in enumerate(m_channels_per_block):
            layer_blocks = []
            for _ in range(n_blocks[channel_indx]):
                layer_blocks += [
                    nn.Conv2d(input_channels, n_out_channels, kernel_size=3, padding=1),
                    nn.ReLU(True),
                ]
                input_channels = n_out_channels
            layer_blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
            if channel_indx == len(m_channels_per_block) - 1:
                layer_blocks.append(nn.AdaptiveAvgPool2d((2, 2)))
            encoder_list.append(nn.Sequential(*layer_blocks))
        rand_tensor = self.get_random_input(image_space.shape)
        representation_size = self._compute_linear_input(encoder_list, rand_tensor)

        classifier = []
        classifier.append(
            nn.Sequential(
                *[
                    nn.Flatten(),
                    nn.Linear(representation_size, 1024),
                    nn.ReLU(True),
                    nn.Dropout(),
                ]
            )
        )
        classifier.append(
            nn.Sequential(*[nn.Linear(1024, 512), nn.ReLU(True), nn.Dropout(),])
        )
        classifier.append(nn.Linear(512, n_classes))
        super(VGG7, self).__init__(encoder_list + classifier, len(encoder_list))
