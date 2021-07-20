from torch import nn as nn
import torch
import random


class Model(nn.Module):
    """Model wrapper for all models 
    """

    def __init__(self, module_list: list, penulimate_layer_indx: int, n_classes: int, bic: bool = False):
        # FIXME put its params inside dataclass
        super(Model, self).__init__()
        self.model = nn.ModuleList(module_list)
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.penulimate_layer_indx = penulimate_layer_indx
        self.alpha = None
        self.manifold_mix = False
        self.n_classes = n_classes
        # Bias control
        self.bic = bic
        if self.bic:
            self.current_task = 0
            self.bic_params = nn.Parameter(
                torch.cat([torch.zeros(1, self.n_classes), torch.ones(1, self.n_classes)]))

    def new_task(self, task_id=None):
        self.bic_params.data = torch.cat([torch.zeros(1, self.n_classes), torch.ones(
            1, self.n_classes)]).to(self.dummy_param.device)  # default to no bias (logit= 1*l +0)

    def __iter__(self):
        """ Returns the Iterator object """
        return iter(self.model)

    def __len__(self):
        return len(self.model)

    def get_penultimate(self, x):
        penultimate_output = self.forward(
            x, to_layer=self.penulimate_layer_indx)
        logits = self.forward(
            penultimate_output, from_layer=self.penulimate_layer_indx + 1
        )
        return penultimate_output, logits

    def forward(self, x, from_layer=None, to_layer=None, bic=True):
        """forward pass supporting getting output from a specifc layer and forward starting from out of another layer

        Args:
            x (tensor): batch of data points
            from_layer (int, optional): index of starting layer for the forward pass. Defaults to None.
            to_layer (int, optional): index of output layer for the forward pass. Defaults to None.
            bic (flag, optional): flag to enable bias control
        Returns:
            (tensor): batch of model's output
        """
        device = self.dummy_param.device
        x = x.to(device)
        # TODO check for compatibility this was necessary for rl setting
        if x.dim() == 3 or x.dim() == 1:
            x = x.unsqueeze(0)
        for layer_indx, module_pt in enumerate(self.model):
            if type(from_layer) == int and layer_indx < from_layer:
                continue
            x = module_pt(x)
            if type(to_layer) == int and to_layer == layer_indx:
                # used to get layer output at a specific manifold layer
                return x

        if self.bic and bic and not(self.training):
            x = self.apply_bic(x)
        return x

    def apply_bic(self, x):
        x *= self.bic_params[1]
        x += self.bic_params[0]
        return x

    def _compute_linear_input(self, module_list, x):
        """computes dynamically size of input features to the fully connected part of the model

        Arguments:
            module_list {list} -- list of  convolutional modules that we want to compute its output size
            x (tensor)  -- input tensor used to get linear output
        Returns:
            int -- output size of the input module list
        """
        model = nn.ModuleList(module_list)
        for module_pt in model:
            x = module_pt(x)
        x = nn.Flatten()(x)
        return x.shape[1]

    def get_random_input(self, input_shape):
        """takes input size and generates a random torch tensor to test the model

        Arguments:
            input_size {int} -- size of the input

        Returns:
            torch.tensor -- random tensor based on the input
        """
        n_channels = input_shape[0]
        input_size = input_shape[1:]
        return torch.rand(1, n_channels, *input_size)

    def get_encoder(self):
        # Fixme debug
        encoder = []
        for layer in self.model[: self.penulimate_layer_indx + 1]:
            encoder.append(layer)
        return encoder

    def get_decoder(self):
        # FIXME debug it
        decoder = []
        for layer in self.model[self.penulimate_layer_indx + 1:]:
            decoder.append(layer)
        return decoder
