""" WIP: Exploring the idea of creating a 'universal encoder' function, that
would create an appropriate model for mapping from any given input space to
output space, given a specified budget (maximum number of network parameters).
"""
import bisect
import math
import warnings
from functools import singledispatch, partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Type, TypeVar)
import itertools
import torch
import numpy as np
from gym import Space, spaces
from gym.spaces.utils import flatdim, flatten, flatten_space, unflatten
from torch import Tensor, nn
from torch.nn.utils import parameters_to_vector
from torchvision import models
from torchvision.models import ResNet
T = TypeVar("T")

# Dict of vision models, mapping from number of parameters to tuples with the
# type of model and the resulting output space.

resnet_output_space = spaces.Box(0., 1., shape=[1000,])

vision_models: Dict[int, Tuple[Type[nn.Module], Space]] = {
    11_689_512: (models.resnet18,  resnet_output_space),
    21_797_672: (models.resnet34,  resnet_output_space),
    25_557_032: (models.resnet50,  resnet_output_space),
    44_549_160: (models.resnet101, resnet_output_space),
    60_192_808: (models.resnet152, resnet_output_space),
}

def n_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

# TODO: Multi-dispatch ? Could be pretty sweet!    

@singledispatch
def create_encoder(input_space: Space,
                   output_space: Space,
                   budget: int = None,
                   **kwargs) -> Tuple[nn.Module, Space]:
    """Creates a neural net encoder mapping from an arbitrary input space to an
    arbitrary output space.

    Parameters
    ----------
    input_space : Space
        Some input space. This can also be structured, i.e. this could be a
        Tuple or Dict space, or an arbitrary nesting of such spaces. 
    output_space : Space
        Some output space. Box is preferable, but this could in essence 
    budget : int, optional
        [description], by default None
    **kwargs : Any parameters / hyper-parameters for each part of the encoder.

    Returns
    -------
    nn.Module
        [description]
    """
    raise NotImplementedError(f"No known encoder for input space {input_space}")


class DictEncoder(nn.Module):
    def __init__(self, encoders: Dict[str, nn.Module]):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
    def forward(self, x: Dict[str, Tensor]):
        return {
            key: self.encoders[key](tensor)
            for key, tensor in x.items()
        }

class TupleEncoder(nn.Module):
    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self.modules = nn.ModuleList(modules)
    def forward(self, x: Sequence[Tensor]):
        assert len(x) == len(self.modules)
        return tuple(self.modules[i](tensor) for i, tensor in enumerate(x))

class LambdaModule(nn.Module):
    def __init__(self, f):
        super().__init__()
        assert callable(f)
        self.f = f

    def forward(self, x):
        return self.f(x) 


@create_encoder.register
def dict_encoder(input_space: spaces.Dict,
                 output_space: Space,
                 budget: int = None,
                 shared_budget: int = None,
                 hidden_dims: int = 512,
                 split: Dict[str, Any] = None,
                 shared: Dict[str, Any] = None,
                 **kwargs) -> nn.Module:
    """ IDEA: Create an encoder for each item in the dict, mapping from the
    corresponding input space to some kind of latent space, and then add a
    flatten/concat layer, then map to the provided output space.
    
    shared_budget: The budget for the shared portion of the network. Must be
    less than the `budget`. If only `budget` is given, the shared budget is set
    to 1/2 of the total budget. 
    
    """
    split = split or {}
    shared = shared or {}
    if kwargs:
        warnings.warn(RuntimeWarning(
            f"Ignoring kwargs {kwargs}! (This acceps 'split' and 'shared' to "
            f"hold the hparams of the split and shared portions of the "
            f"encoder)."
        ))

    total_input_dims = flatdim(input_space)
    total_output_dims = flatdim(output_space)
    n_inputs = len(input_space.spaces)

    if not isinstance(output_space, spaces.Box):
        raise NotImplementedError("Only support Box output spaces for now.")

    split_budget: Optional[int] = None
    shared_budget: Optional[int] = None

    if budget is not None:
        if shared_budget is None:
            shared_budget = budget // 2
        split_budget = budget - shared_budget
    
    encoders: Dict[str, nn.Module] = {}
    latent_spaces: Dict[str, Space] = {}
    
    for key, subspace in input_space.spaces.items():
        dimension_input_dim = flatdim(subspace)
        # TODO: Each output will be a Box for now, and each dimension will have
        # a number of 'dedicated' features in the 'output space' that will be
        # proportional to their size in the input space. 
        dimension_output_dim = round(dimension_input_dim / total_input_dims * hidden_dims)
        dimension_output_dim = max(dimension_output_dim, 1)
        dimension_output_dim = min(dimension_output_dim, total_output_dims - (n_inputs-1))
        assert 0 < dimension_output_dim < total_output_dims

        dimension_latent_space: Space = spaces.Box(0, 1, shape=[dimension_output_dim])
        latent_spaces[key] = dimension_latent_space
        
        # The 'budget', in number of parameters, that gets allotted for the
        # encoding of this dimension.
        dimension_budget = None
        if split_budget is not None:
            # The dimension gets a portion of the budget based on the proportion
            # of its input space compared to the total.
            dimension_budget = round(dimension_input_dim / total_input_dims * split_budget)
        else:
            dimension_budget = None

        encoders[key] = create_encoder(
            subspace,
            output_space=dimension_latent_space,
            budget=dimension_budget,
            **split.get(key, {})
        )

    # Encoder that processes each input separately and produces a "latent space"
    # for each input dimension. (dict input, dict output).
    split_encoders_module = DictEncoder(encoders)
    actual_split_params = n_parameters(split_encoders_module)
    
    if split_budget is not None:
        if actual_split_params > split_budget:
            warnings.warn(RuntimeWarning(
                f"The budget for the 'split' portion of the encoder was "
                f"{split_budget} parameters, but somehow the constructed "
                f"module has {actual_split_params} parameters!"
            ))

    # Operation that 'concatenates' all the hidden spaces together.
    concat_layer = LambdaModule(f=lambda d: torch.cat(list(d.values()), dim=-1))
    latent_dims = sum(map(flatdim, latent_spaces.values()))
    fused_latent_space = spaces.Box(-np.inf, np.inf, shape=[latent_dims],)
    
    assert latent_dims == hidden_dims, "The sum of latent spaces didn't equal the prescribed hidden dims?"
    
    shared_module = create_encoder(
        fused_latent_space,
        output_space=output_space,
        budget=budget,
        **shared,
    )
    return nn.Sequential(
        split_encoders_module,
        concat_layer,
        shared_module,
    )

@create_encoder.register
def tuple_encoder(input_space: spaces.Tuple,
                 output_space: Space,
                 budget: int = None,
                 **kwargs) -> nn.Module:
    """ IDEA: Create an encoder for each item in the tuple, mapping from the
    corresponding input space to some kind of latent space, and then add a
    flatten/concat layer, then map to the provided output space.
    
    shared_budget: The budget for the shared portion of the network. Must be
    less than the `budget`. If only `budget` is given, the shared budget is set
    to 1/2 of the total budget. 
    """
    # 'defer' to the dict_encoder method, with the indices as the keys. 
    return create_encoder(
        input_space=spaces.Dict({
            str(i): subspace for i, subspace in
            enumerate(input_space.spaces)
        }),
        output_space=output_space,
        budget=budget,
        **kwargs
    )

def is_image_space(space: spaces.Box) -> bool:
    shape = space.shape
    return len(shape) == 3 and (
        shape[0] == shape[1] and shape[2] in {1, 3} or
        shape[1] == shape[2] and shape[0] in {1, 3}
    )

def get_vision_model(input_space: Space, budget: int = None) -> Tuple[Type[nn.Module], Space]:
    """ Returns the biggest vision model (Resnet for now) that fits within the
    given budget (in number of parameters).
    """
    assert is_image_space(input_space)
    n_parameters = sorted(vision_models.keys())
    if budget is None:
        # If we have unlimited budget, use the biggest vision model available.
        return vision_models[n_parameters[-1]]
    # Get the largest model with fewer than `budget` parameters.
    return vision_models[n_parameters[bisect.bisect_left(n_parameters, budget)]]

# TODO: Idea: Create an Image() space, as a subclass of spaces.Box
Image = spaces.Box

# @create_encoder.register(Image)
def image_encoder(input_space: Image, output_space: Space, budget: int = None, **kwargs) -> nn.Module:
    # If we are on a budget, then use the largest vision model that fits our
    # budget.
    vision_model, hidden_space = get_vision_model(input_space, budget=budget)
    if hidden_space == output_space:
        return vision_model
    
    encoder = vision_model(**kwargs)
    if budget is not None:
        budget -= n_parameters(encoder)
    
    return nn.Sequential(
        encoder,
        create_encoder(hidden_space, output_space, budget=budget)
    )

def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class MLP(nn.Sequential):
    def __init__(self,
                 input_dims: int,
                 *hidden_dims: int,
                 output_dims: int,
                 activation_type: Optional[Type[nn.Module]] = torch.nn.ReLU):
        self.input_dims = input_dims
        self.hidden_dims: List[int] = list(hidden_dims)
        self.output_dims = output_dims
        self.activation_type = activation_type
        layers: List[nn.Module] = []
        units = [input_dims, *hidden_dims, output_dims]
        for i, (in_features, out_features) in enumerate(pairwise(units)):
            layers.append(nn.Linear(in_features, out_features))
            if activation_type and i != len(units) - 2:
                # Add an activation layer after all but the last layer.
                layers.append(activation_type())
        super().__init__(*layers)



@create_encoder.register
def box_encoder(input_space: spaces.Box,
                output_space: Space,
                budget: int = None,
                hidden_dims: List[int] = None,
                **kwargs) -> nn.Module:
    input_dims = flatdim(input_space)
    output_dims = flatdim(output_space)
    
    assert isinstance(output_space, spaces.Box), "only support box output shape for now."
    
    if is_image_space(input_space):
        return image_encoder(input_space, output_space, budget=budget, **kwargs)
        
    if hidden_dims is None:
        if budget is not None:
            # There are, in total, this many parameters, as a function of the input
            # size, hidden size, output size, and number of layers.
            # Would be cool if we could determine the hidden_dims given the budget.  
            # n_params = (
            #     input_dims * hidden_dims + hidden_dims + # first dense layer
            #     (hidden_dims * hidden_dims + hidden_dims) * (n_layers - 2) + # Hidden layers
            #     hidden_dims * output_dims + hidden_dims # Output layer
            # )
            n_layers = 3
            hidden_dim = round(math.sqrt(budget // n_layers))
            hidden_dims = [
                hidden_dim for _ in range(n_layers)
            ]
        else:
            hidden_dims = [
                64, 64, 64,
            ]

    return MLP(
        input_dims,
        *hidden_dims,
        output_dims=output_dims,
    )
    # vision_model, output_space = 


class Tile(nn.Module):
    def __init__(self, output_dims: int):
        super().__init__()
        self.output_dims = output_dims

    def forward(self, x: Tensor):
        assert x.ndim == 1
        x = x.reshape([-1, 1])
        return x.expand([-1, self.output_dims])
        assert False, (x, self.output_dims)

@create_encoder.register
def discrete_encoder(input_space: spaces.Discrete, output_space: Space, budget: int = None, **kwargs) -> nn.Module:
    # Just tile / copy the input value into a tensor.
    assert isinstance(output_space, spaces.Box), "Only support box output space for now."
    assert output_space.dtype == np.float32, output_space.dtype
    output_dims = flatdim(output_space)
    return Tile(output_dims)
    return LambdaModule(lambda v: torch.empty(output_space.shape, dtype=torch.float).fill_(v))        


@create_encoder.register
def multidiscrete_encoder(input_space: spaces.MultiDiscrete, output_space: Space, budget: int = None) -> nn.Module:
    # if input_space.shape[0]
    input_dims = flatdim(input_space)
    output_dims = flatdim(output_space)
    assert isinstance(output_space, spaces.Box), "Only support Box output spaces for now."
    return nn.Sequential(
        LambdaModule(lambda v: v.reshape([-1, input_dims]).to(torch.float)),
        nn.Linear(input_dims, output_dims),
    )
