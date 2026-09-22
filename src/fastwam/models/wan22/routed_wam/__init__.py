"""RoutedWAM: generative, routed and interface-distilled imagination for FastWAM.

This package is additive: it installs itself on top of an existing
:class:`~fastwam.models.wan22.dream_fastwam.model.DreamFastWAM` and does not
modify any shipped module.
"""

from .generative_dream import DreamTargetEncoder, GenerativeDreamExpert
from .interface_distill import InterfaceDistillConfig, InterfaceDistiller
from .model import RoutedWAM
from .mot import RoutedMoT
from .router import ImaginationRouter, RouterConfig, build_group_ids

__all__ = [
    "DreamTargetEncoder",
    "GenerativeDreamExpert",
    "ImaginationRouter",
    "InterfaceDistillConfig",
    "InterfaceDistiller",
    "RoutedMoT",
    "RoutedWAM",
    "RouterConfig",
    "build_group_ids",
]
