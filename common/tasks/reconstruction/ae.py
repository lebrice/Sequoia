""" Defines an Auto-Encoder-based Auxiliary task.
"""
from typing import ClassVar, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from common.loss import Loss
from common.tasks.auxiliary_task import AuxiliaryTask

from .decoder_for_dataset import get_decoder_class_for_dataset


class AEReconstructionTask(AuxiliaryTask):
    """ Task that adds the AE loss (reconstruction loss). 
    
    Uses the feature extractor (`encoder`) of the parent model as the encoder of
    an AE. Contains trainable `decoder` module, which is
    used to get the AE loss to train the feature extractor with.
    """
    name: ClassVar[str] = "ae"

    def __init__(self,
                 coefficient: float = None,
                 options: AuxiliaryTask.Options = None):
        super().__init__(coefficient=coefficient, options=options)
        self.loss = nn.MSELoss(reduction="sum")
        
        # BUG: The decoder for mnist has output shape of [1, 28, 28], but the
        # transforms 'fix' that shape to be [3, 28, 28].
        # Therefore: TODO: Should we adapt the output shape of the decoder
        # depending on the shape of the input?
        self.decoder: Optional[nn.Module] = None

    def create_decoder(self, input_shape: Union[torch.Size, Tuple[int, ...]]) -> nn.Module:
        """ Creates a decoder to reconstruct the input from the hidden vectors.
        """
        if len(input_shape) == 4:
            # discard the batch dimension.
            input_shape = input_shape[1:]
        # At the moment we have a 'fixed' set of image sizes (28, 32, 224, iirc)
        # and we just use the decoder type for the given dataset. 
        # TODO: Create the decoder dynamically, depending on the required shape.
        decoder_class = get_decoder_class_for_dataset(input_shape)
        decoder: nn.Module = decoder_class(
            code_size=AuxiliaryTask.hidden_size,
        )
        decoder = decoder.to(self.device)
        return decoder

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor=None, y: Tensor=None) -> Loss:
        z = h_x.view([h_x.shape[0], -1])
        if self.decoder is None or self.decoder.output_shape != x.shape:
            self.decoder = self.create_decoder(x.shape)
        x_hat = self.decoder(z)
        assert x_hat.shape == x.shape, (
            f"reconstructed x should have same shape as original x! "
            f"({x_hat.shape} != {x.shape})"
        )
        recon_loss = self.reconstruction_loss(x_hat, x)
        loss_info = Loss(name=self.name, loss=recon_loss)
        return loss_info

    def forward(self, h_x: Tensor) -> Tensor:  # type: ignore
        z = h_x.view([h_x.shape[0], -1])
        x_hat = self.decoder(z)
        return x_hat

    def reconstruct(self, x: Tensor) -> Tensor:
        h_x = self.encode(x)
        x_hat = self.forward(h_x)
        return x_hat.view(x.shape)

    def reconstruction_loss(self, recon_x: Tensor, x: Tensor) -> Tensor:
        return self.loss(recon_x, x)
