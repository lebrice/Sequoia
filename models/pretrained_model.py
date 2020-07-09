import warnings
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar, Union, Tuple

import torch
from simple_parsing import choice
from torch import nn


def get_pretrained_encoder(hidden_size: int,
                           encoder_model: Callable,
                           pretrained: bool=False,
                           freeze_pretrained_weights: bool=False) -> nn.Module:
    """Returns a pretrained encoder on ImageNet from `torchvision.models`

    Will try to replace the classification layer/block with a
    `nn.Linear(<h>, hidden_size)`, where <h> corresponds to the hidden size of
    the model. This last layer will always be trainable, even if
    `freeze_pretrained_weights` is True.

    Args:
        hidden_size (int): The hidden size of the resulting model.
        encoder_model (Callable): Which encoder model to use. Should usually be
            one of the models in the `torchvision.models` module.
        pretrained (bool, optional): Wether to try and download the pretrained
            weights. Defaults to True.
        freeze_pretrained_weights (bool, optional): Wether the pretrained
            (downloaded) weights should be frozen. Has no effect when
            `pretrained` is False. Defaults to False.
    Returns:
        nn.Module: the pretrained encoder, with the classification head removed.
    """

    print("Using encoder model", encoder_model.__name__,
          "pretrained: ", pretrained,
          "freezing the pretrained weights: ", freeze_pretrained_weights)
    encoder = encoder_model(pretrained=pretrained)

    if pretrained and freeze_pretrained_weights:
        # Fix the parameters of the model.
        for param in encoder.parameters():
            param.requires_grad = False
    
    # We want to replace the last layer (the classification layer) with a
    # projection from their hidden space dimension to ours.
    new_classifier: Optional[nn.Linear] = None
    for attr in ["classifier", "fc"]:
        if hasattr(encoder, attr):
            classifier: Union[nn.Sequential, nn.Linear] = getattr(encoder, attr)
            # Get the number of input features.
            if isinstance(classifier, nn.Linear):
                in_features = classifier.in_features
                new_classifier = nn.Linear(
                    in_features=classifier.in_features,
                    out_features=hidden_size
                )
            elif isinstance(classifier, nn.Sequential):
                # if there is a classifier "block", get the number of
                # features from the first encountered dense layer.
                for layer in classifier.children():
                    if isinstance(layer, nn.Linear):
                        new_classifier = nn.Linear(layer.in_features, hidden_size)
                        break
            break

    if new_classifier:
        setattr(encoder, attr, new_classifier)
        print(f"Replaced the attribute '{attr}' of the {encoder_model.__name__} model with a new classifier: {new_classifier}")
    else:
        warnings.warn(
            f"Can't detect the hidden size of the model '{encoder_model.__name__}'!"
            f" (last layer is :{classifier}).\n"
            "Returning the model as-is."
        )
    return encoder
    
