"""
Vendor Integrations for Surgical Robotics Systems.

Provides compatibility layers for major surgical robot manufacturers:
- Intuitive Surgical (Da Vinci Xi/X/SP)
- Medtronic (Hugo RAS, Mazor X)
- Stryker (Mako)
- Zimmer Biomet (ROSA)
- Smith+Nephew (CORI)
- Brainlab
"""

from surgical_robotics.vendors.base import (
    VendorInterface,
    VendorCapabilities,
    ConnectionStatus,
    VendorConfig,
)
from surgical_robotics.vendors.intuitive import IntuitiveDaVinciInterface
from surgical_robotics.vendors.medtronic import MedtronicHugoInterface, MedtronicMazorInterface
from surgical_robotics.vendors.stryker import StrykerMakoInterface
from surgical_robotics.vendors.zimmer import ZimmerRosaInterface
from surgical_robotics.vendors.smith_nephew import SmithNephewCoriInterface

__all__ = [
    "VendorInterface",
    "VendorCapabilities",
    "ConnectionStatus",
    "VendorConfig",
    "IntuitiveDaVinciInterface",
    "MedtronicHugoInterface",
    "MedtronicMazorInterface",
    "StrykerMakoInterface",
    "ZimmerRosaInterface",
    "SmithNephewCoriInterface",
]
