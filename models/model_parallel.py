import torch
from torch import device
import numpy as np
import warnings
from torchvision.models.resnet import ResNet, Bottleneck
from torch import nn
from typing import List, Tuple


class ModelParallelResNet50(ResNet):  
    def __init__(self, num_classes=1000, pretrained=False, *args, **kwargs):
        #assert torch.cuda.device_count() >= len(np.unique(device))
        warnings.warn(UserWarning(
                f"Parallel models are not pretrained for now. Argument 'pretrained' is ignored here. "
            ))
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes,  *args, **kwargs)
        
        self.device = (device(type='cuda', index=0), device(type='cuda', index=1))
        
        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        )#.to(self.device[0])

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        )#.to(self.device[1])

        #self.fc.to(self.device[1])

    def forward(self, x):
        x = self.seq2(self.seq1(x).to(self.device[1]))
        return self.fc(x.view(x.size(0), -1))
    
    def to(self, device: Tuple = (device(type='cuda', index=0), device(type='cuda', index=1))):
            self.device = device
            self.seq1.to(device[0])
            self.seq2.to(device[1])
            self.fc.to(device[1])




class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=0.5, *args, **kwargs):       
        super(PipelineParallelResNet50, self).__init__(num_classes=1000, *args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        #print(x.shape)
        splits = iter(x.split( int(len(x) * self.split_size), dim=0))
        s_next = next(splits)
        #print(s_next.shape)
        s_prev = self.seq1(s_next).to(self.device[1])
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to(self.device[1])

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)