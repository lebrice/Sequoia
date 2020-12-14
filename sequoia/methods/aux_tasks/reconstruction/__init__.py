""" Auxiliary tasks based on reconstructing an input given a hidden vector.

TODO: Add some denoising autoencoders maybe as a reconstruction task?
"""
from .ae import AEReconstructionTask
from .decoders import CifarDecoder, MnistDecoder
from .decoder_for_dataset import get_decoder_class_for_dataset
from .vae import VAEReconstructionTask
