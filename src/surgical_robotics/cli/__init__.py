"""
Command Line Interface for Surgical Robotics.

Provides:
- Robot configuration and setup
- Calibration utilities
- Simulation runner
- System diagnostics
"""

from surgical_robotics.cli.commands import (
    main,
    run_simulation,
    calibrate,
    diagnose,
)
from surgical_robotics.cli.config import (
    Configuration,
    RobotConfig,
    SafetyConfig,
    load_config,
    save_config,
)

__all__ = [
    "main",
    "run_simulation",
    "calibrate",
    "diagnose",
    "Configuration",
    "RobotConfig",
    "SafetyConfig",
    "load_config",
    "save_config",
]
