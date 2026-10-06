"""RoutedWAM: generative, routed and interface-distilled imagination for FastWAM.

This package installs itself on top of an existing
:class:`~fastwam.models.wan22.dream_fastwam.model.DreamFastWAM` and uses its
action-cache adapter hook for semantic routing.
"""

from .generative_dream import DreamTargetEncoder, GenerativeDreamExpert
from .interface_distill import InterfaceDistillConfig, InterfaceDistiller
from .model import RoutedWAM
from .mot import RoutedMoT
from .router import DynamicFeatureRouter, ImaginationRouter, RouterConfig, build_group_ids

__all__ = [
    "DynamicFeatureRouter",
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
