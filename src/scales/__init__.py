"""
SCALeS: generation, modelling and simulation of realistic very large scale supply chain instances.

The pipeline has four stages; each can be used on its own and writes its default output to its own folder
(see scales.config):

    1. SupplyChainBackbone    layered Chung-Lu backbone graph            -> networks/backbones/
    2. ScbConverter           static quantitative supply chain (JSON)    -> supply_chains/, networks/supply_chains/
    3. SupplyChain            dynamic network flow model, solved by LP   -> networks/dynamic/, results/
    4. SupplyChainAnalyser    perturbation (resilience) analysis         -> results/perturbation/
"""

from . import config
from .backbone import SupplyChainBackbone
from .conversion import ScbConverter
from .model import SupplyChain
from .analysis import SupplyChainPerturber, SupplyChainAnalyser, find_bottlenecks, print_bottlenecks

__version__ = "0.2.0"

__all__ = [
    "config",
    "SupplyChainBackbone",
    "ScbConverter",
    "SupplyChain",
    "SupplyChainPerturber",
    "SupplyChainAnalyser",
    "find_bottlenecks",
    "print_bottlenecks",
]
