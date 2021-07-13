""" Sequoia - The Research Tree """
from .settings import *
from .experiments import Experiment

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
