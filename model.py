
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
import torch.functional as F
from torch import Tensor, nn

from classifier import Classifier


class SemiSupervisedModel(Classifier):
    def __init__(self, num_labels: int = 10):
        super().__init__()
        self.num_labels: int = num_labels

        # self.features: models.ResNet = models.resnet18(pretrained=False)
        # self.features.fc = nn.Sequential()
        self.features_dim: int = 512  # TODO: figure this out programmatically?
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(3*128*128, self.features_dim)
        )
        self.classifier: nn.Module = nn.Sequential(
            nn.Linear(self.features_dim, self.features_dim//2),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(self.features_dim//2, self.num_labels)
        )

    def preprocess_inputs(self, x: Tensor) -> Tensor:
        return x.view([x.shape[0], -1])

    def extract_features(self, x: Tensor) -> Tensor:
        return self.feature_extractor(x)

    def loss(self, x: Tensor, y: Tensor=None) -> Tensor:
        x = self.preprocess_inputs(x)
        h_x = self.(x)
        y_pred = self.classifier(h_x)
        loss = 0.
        if y_true is not None:
            loss += self.supervised_loss()

    def supervised_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        pass

    def unsupervised_loss(self, x: Tensor) -> Tensor:
        pass

    def semisupervised_loss(self, x: Tensor, *args: Any) -> Tensor:
        pass

    
m = SemiSupervisedModel()
img = torch.randn(1, 3, 128, 128)
print(m(img))
