"""
Surgical Robotics Control Systems

A comprehensive framework for surgical robot control including:
- Da Vinci-Class multi-arm systems with haptic feedback
- Neurosurgical robots with sub-millimeter precision
- Orthopedic robots with force feedback and bone cutting
- Vendor integrations (Intuitive, Medtronic, Stryker, Zimmer, Smith+Nephew)
- Communication protocols (ROS2, DICOM, HL7/FHIR)
- Advanced control (MPC, impedance, reinforcement learning)
- Simulation environment with physics and tissue modeling
- Workflow and state machine management
- Telemetry, logging, and monitoring
- Medical standards compliance (IEC 62304, IEC 60601, FDA 21 CFR 820)
"""

__version__ = "0.2.0"

# Core modules
from surgical_robotics.core.base import SurgicalRobot, RobotArm, EndEffector, Pose
from surgical_robotics.core.kinematics import ForwardKinematics, InverseKinematics
from surgical_robotics.core.safety import SafetyController, CollisionDetector

__all__ = [
    # Core
    "SurgicalRobot",
    "RobotArm",
    "EndEffector",
    "Pose",
    "ForwardKinematics",
    "InverseKinematics",
    "SafetyController",
    "CollisionDetector",
]

# Lazy imports for submodules to avoid heavy dependencies at startup
def __getattr__(name):
    """Lazy loading of submodules."""
    submodules = {
        "davinci": "surgical_robotics.davinci",
        "neurosurgical": "surgical_robotics.neurosurgical",
        "orthopedic": "surgical_robotics.orthopedic",
        "vendors": "surgical_robotics.vendors",
        "communication": "surgical_robotics.communication",
        "control": "surgical_robotics.control",
        "simulation": "surgical_robotics.simulation",
        "workflow": "surgical_robotics.workflow",
        "telemetry": "surgical_robotics.telemetry",
        "compliance": "surgical_robotics.compliance",
        "cli": "surgical_robotics.cli",
    }

    if name in submodules:
        import importlib
        return importlib.import_module(submodules[name])

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
