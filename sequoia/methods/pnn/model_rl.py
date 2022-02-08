from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .layers import PNNConvLayer, PNNLinearBlock


class PnnA2CAgent(nn.Module):
    """
    @article{rusu2016progressive,
      title={Progressive neural networks},
      author={Rusu, Andrei A and Rabinowitz, Neil C and Desjardins, Guillaume and Soyer, Hubert and Kirkpatrick, James and Kavukcuoglu, Koray and Pascanu, Razvan and Hadsell, Raia},
      journal={arXiv preprint arXiv:1606.04671},
      year={2016}
    }
    """

    def __init__(self, arch="mlp", hidden_size=256):
        super(PnnA2CAgent, self).__init__()
        self.columns_actor = nn.ModuleList([])
        self.columns_critic = nn.ModuleList([])
        self.columns_conv = nn.ModuleList([])
        self.arch = arch
        self.hidden_size = hidden_size
        # TODO: This doesn't take the observation space into account at all!
        # Only works for Pixel Cartpole at the moment.
        # Original size 3 x 400 x 600
        self.transformation = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def forward(self, observations):
        assert (
            self.columns_actor
        ), "PNN should at least have one column (missing call to `new_task` ?)"
        t = observations.task_labels

        if self.arch == "mlp":
            x = torch.from_numpy(observations.x).unsqueeze(0).float()
            inputs_critic = [c[1](c[0](x)) for c in self.columns_critic]
            inputs_actor = [c[1](c[0](x)) for c in self.columns_actor]

            outputs_critic = []
            outputs_actor = []
            for i, column in enumerate(self.columns_critic):
                outputs_critic.append(column[2](inputs_critic[: i + 1]))
                outputs_actor.append(self.columns_actor[i][2](inputs_actor[: i + 1]))

            ind_depth = 3

        else:
            x = self.transfor_img(observations.x).unsqueeze(0).float()
            inputs = [c[1](c[0](x)) for c in self.columns_conv]

            outputs = []
            for i, column in enumerate(self.columns_conv):
                outputs.append(column[3](column[2](inputs[: i + 1])))

            inputs = outputs
            outputs = []
            for i, column in enumerate(self.columns_conv):
                outputs.append(column[5](column[4](inputs[: i + 1])))

            inputs_critic = [c[6](outputs[i]).view(1, -1) for i, c in enumerate(self.columns_conv)]
            inputs_actor = inputs_critic[:]

            outputs_critic = []
            outputs_actor = []
            for i, column in enumerate(self.columns_critic):
                outputs_critic.append(column[0](inputs_critic[: i + 1]))
                outputs_actor.append(self.columns_actor[i][0](inputs_actor[: i + 1]))

            ind_depth = 1

        critic = []
        for i, column in enumerate(self.columns_critic):
            critic.append(column[ind_depth](outputs_critic[i]))

        actor = []
        for i, column in enumerate(self.columns_actor):
            actor.append(F.softmax(column[ind_depth](outputs_actor[i]), dim=1))

        return critic[t], actor[t]

    def new_task(self, device, num_inputs, num_actions=5):
        task_id = len(self.columns_actor)

        if self.arch == "conv":
            sizes = [num_inputs, 32, 64, self.hidden_size]
            modules_conv = nn.Sequential()

            modules_conv.add_module("Conv1", PNNConvLayer(task_id, 0, sizes[0], sizes[1]))
            modules_conv.add_module("MaxPool1", nn.MaxPool2d(3))
            modules_conv.add_module("Conv2", PNNConvLayer(task_id, 1, sizes[1], sizes[2]))
            modules_conv.add_module("MaxPool2", nn.MaxPool2d(3))
            modules_conv.add_module("Conv3", PNNConvLayer(task_id, 2, sizes[2], sizes[3]))
            modules_conv.add_module("MaxPool3", nn.MaxPool2d(3))
            modules_conv.add_module("globavgpool2d", nn.AdaptiveAvgPool2d((1, 1)))
            self.columns_conv.append(modules_conv)

        modules_actor = nn.Sequential()
        modules_critic = nn.Sequential()

        if self.arch == "mlp":
            modules_actor.add_module("linAc1", nn.Linear(num_inputs, self.hidden_size))
            modules_actor.add_module("relAc", nn.ReLU(inplace=True))
        modules_actor.add_module(
            "linAc2", PNNLinearBlock(task_id, 1, self.hidden_size, self.hidden_size)
        )
        modules_actor.add_module("linAc3", nn.Linear(self.hidden_size, num_actions))

        if self.arch == "mlp":
            modules_critic.add_module("linCr1", nn.Linear(num_inputs, self.hidden_size))
            modules_critic.add_module("relCr", nn.ReLU(inplace=True))
        modules_critic.add_module(
            "linCr2", PNNLinearBlock(task_id, 1, self.hidden_size, self.hidden_size)
        )
        modules_critic.add_module("linCr3", nn.Linear(self.hidden_size, 1))

        self.columns_actor.append(modules_actor)
        self.columns_critic.append(modules_critic)

        print("Add column of the new task")

    def unfreeze_columns(self):
        for i, c in enumerate(self.columns_actor):
            for params in c.parameters():
                params.requires_grad = True

            for params in self.columns_critic[i].parameters():
                params.requires_grad = True

        for i, c in enumerate(self.columns_conv):
            for params in c.parameters():
                params.requires_grad = True

    def freeze_columns(self, skip: List[int] = None):
        if skip is None:
            skip = []

        self.unfreeze_columns()

        for i, c in enumerate(self.columns_actor):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

                for params in self.columns_critic[i].parameters():
                    params.requires_grad = False

        for i, c in enumerate(self.columns_conv):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

        print("Freeze columns from previous tasks")

    def parameters(self, task_id):
        param = []
        for p in self.columns_critic[task_id].parameters():
            param.append(p)
        for p in self.columns_actor[task_id].parameters():
            param.append(p)

        if len(self.columns_conv) > 0:
            for p in self.columns_conv[task_id].parameters():
                param.append(p)

        return param

    def transfor_img(self, img):
        return self.transformation(img)
        # return lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.
