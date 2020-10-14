import warnings
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar, Union

import torch
from simple_parsing import choice
from torch import nn
from utils.logging_utils import get_logger
logger = get_logger(__file__)
from typing import Tuple

def get_pretrained_encoder(encoder_model: Callable,
                           pretrained: bool=True,
                           freeze_pretrained_weights: bool=False,
                           new_hidden_size: int=None,
                           ) -> Tuple[nn.Module, int]:
    """Returns a pretrained encoder on ImageNet from `torchvision.models`

    If `new_hidden_size` is True, will try to replace the classification layer
    block with a `nn.Linear(<h>, new_hidden_size)`, where <h> corresponds to the
    hidden size of the model. This last layer will always be trainable, even if
    `freeze_pretrained_weights` is True.

    Args:
        encoder_model (Callable): Which encoder model to use. Should usually be
            one of the models in the `torchvision.models` module.
        pretrained (bool, optional): Wether to try and download the pretrained
            weights. Defaults to True.
        freeze_pretrained_weights (bool, optional): Wether the pretrained
            (downloaded) weights should be frozen. Has no effect when
            `pretrained` is False. Defaults to False.
        new_hidden_size (int): The hidden size of the resulting model.
    
    Returns:
        Tuple[nn.Module, int]: the pretrained encoder, with the classification
            head removed, and the resulting output size (hidden dims)
    """

    logger.debug(f"Using encoder model {encoder_model.__name__}")
    logger.debug(f"pretrained: {pretrained}")
    logger.debug(f"freezing the pretrained weights: {freeze_pretrained_weights}")

    encoder = encoder_model(pretrained=pretrained)

    if pretrained and freeze_pretrained_weights:
        # Fix the parameters of the model.
        for param in encoder.parameters():
            param.requires_grad = False

    replace_classifier = new_hidden_size is not None
    # We want to replace the last layer (the classification layer) with a
    # projection from their hidden space dimension to ours.
    new_classifier: Optional[nn.Linear] = None
    if not replace_classifier:
        # We will create the 'new classifier' but then not add it.
        # this allows us to also get the 'hidden_size' of the resulting encoder.
        new_hidden_size = 1

    for attr in ["classifier", "fc"]:
        if hasattr(encoder, attr):
            classifier: Union[nn.Sequential, nn.Linear] = getattr(encoder, attr)
            new_classifier: Optional[nn.Linear] = None
            
            # Get the number of input features.
            if isinstance(classifier, nn.Linear):
                new_classifier = nn.Linear(
                    in_features=classifier.in_features,
                    out_features=new_hidden_size
                )
            elif isinstance(classifier, nn.Sequential):
                # if there is a classifier "block", get the number of
                # features from the first encountered dense layer.
                for layer in classifier.children():
                    if isinstance(layer, nn.Linear):
                        new_classifier = nn.Linear(layer.in_features, new_hidden_size)
                        break
            break

    if new_classifier is None:
        raise RuntimeError(
            f"Can't detect the hidden size of the model '{encoder_model.__name__}'!"
            f" (last layer is :{classifier}).\n"
        )

    if not replace_classifier:
        new_hidden_size = new_classifier.in_features
        new_classifier = nn.Sequential()
    else:
        logger.debug(
            f"Replacing the attribute '{attr}' of the "
            f"{encoder_model.__name__} model with a new classifier: "
            f"{new_classifier}"
        )
    setattr(encoder, attr, new_classifier)
    return encoder, new_hidden_size
    
