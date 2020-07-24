"""
TODO: Migrate the addons to Pytorch-Lightning, maybe in the form of callbacks
or as optional extensions to be added to Classifier?
"""

from .knn_callback import KnnCallback
from .vae_callback import SaveVaeSamplesCallback