""" Auxiliary tasks based on reconstructing an input given a hidden vector.

TODO: Add some denoising autoencoders maybe as a reconstruction task?
"""
from .ae import AEReconstructionTask
from .decoder_for_dataset import get_decoder_class_for_dataset
from .decoders import CifarDecoder, MnistDecoder
from .vae import VAEReconstructionTask
