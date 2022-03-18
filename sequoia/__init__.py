""" Sequoia - The Research Tree """
from ._version import get_versions
from .settings import Environment, Method, Setting

# from .experiments import Experiment

__version__ = get_versions()["version"]
del get_versions
