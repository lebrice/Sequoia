""" Adapters for Avalanche Strategies, so they can be used as Methods in Sequoia.

See the Avalanche repo for more info: https://github.com/ContinualAI/avalanche
"""

from .base import AvalancheMethod
from .agem import AGEMMethod
from .ar1 import AR1Method
from .cwr_star import CWRStarMethod
from .ewc import EWCMethod
from .gem import GEMMethod

# Still quite buggy, needs to be fixed on the avalanche side.
from .gdumb import GDumbMethod
from .lwf import LwFMethod
from .naive import NaiveMethod
from .replay import ReplayMethod
from .synaptic_intelligence import SynapticIntelligenceMethod
