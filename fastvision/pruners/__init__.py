from .aot import OTMerger
from .base import Pruner, resolve_keep
from .baselines import RandomPruner, TopNormPruner, UniformPruner
from .divprune import DivPrune
from .fastv import FastVPruner
from .tome import ToMeMerger

PRUNERS = {
    "divprune": DivPrune,
    "tome": ToMeMerger,
    "fastv": FastVPruner,
    "aot": OTMerger,
    "random": RandomPruner,
    "uniform": UniformPruner,
    "topnorm": TopNormPruner,
}

__all__ = [
    "Pruner",
    "resolve_keep",
    "DivPrune",
    "ToMeMerger",
    "FastVPruner",
    "OTMerger",
    "RandomPruner",
    "UniformPruner",
    "TopNormPruner",
    "PRUNERS",
]
