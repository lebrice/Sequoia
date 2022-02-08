import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentType, HParams


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stack_idx: int, block_idx: int):
        super(BasicBlock, self).__init__()

        self.is_downsample_layer = stack_idx > 1 and block_idx == 1
        stride = 2 if self.is_downsample_layer else 1

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = (
            nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
            if self.is_downsample_layer
            else None
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        inplanes: int = 3,
        depth: int = 2,
        width: int = 1,
        stack_planes=[64, 128, 256, 512],
    ):
        super(ResNet, self).__init__()

        prev_filters = stack_planes[0]

        self.conv1 = nn.Conv2d(
            inplanes, prev_filters * width, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(prev_filters * width)
        self.relu = nn.ReLU(inplace=True)

        stacks = []
        for stack_idx, filters in enumerate(stack_planes):
            stacks.append(
                self._make_stack(prev_filters * width, filters * width, depth, stack_idx + 1)
            )
            prev_filters = filters
        self.stacks = nn.Sequential(*stacks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stack_planes[-1] * width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stack(self, inplanes, planes, depth, stack_idx: int):
        layers = []
        layers.append(BasicBlock(inplanes, planes, stack_idx, 1))
        for block_idx in range(2, depth + 1):
            layers.append(BasicBlock(planes, planes, stack_idx, block_idx))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stacks(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Encoder(torch.nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        self.resnet = ResNet(hp.cifar, 3, hp.resnet_depth, hp.resnet_width, hp.resnet_stacks)
        self.resnet.fc = torch.nn.Identity()

    def forward(self, x):
        return self.resnet(x)


class Projector(torch.nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        self.d1 = torch.nn.Linear(hp.repr_dim, hp.repr_dim)
        self.d2 = torch.nn.Linear(hp.repr_dim, hp.proj_dim)

    def forward(self, x):
        x = F.relu(self.d1(x))
        return self.d2(x)


class Classifier(torch.nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        if hp.experiment == ExperimentType.SUCCESSIVE:
            init_dim = hp.proj_dim
        else:
            init_dim = hp.repr_dim
        self.d1 = torch.nn.Linear(init_dim, init_dim)
        self.d2 = torch.nn.Linear(init_dim, hp.cifar)

    def forward(self, x):
        return self.d2(F.relu(self.d1(x)))


# https://github.com/facebookresearch/moco/blob/master/moco/builder.py
class MoCo(nn.Module):
    def __init__(self, hp: HParams, K=512 * 20, m=0.999, T=0.07):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        dim = hp.proj_dim

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = Encoder(hp)
        self.encoder_k = Encoder(hp)
        self.projector_q = Projector(hp)
        self.projector_k = Projector(hp)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.projector_q(self.encoder_q(im_q))  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.projector_k(self.encoder_k(im_k))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
