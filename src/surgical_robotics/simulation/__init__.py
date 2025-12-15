"""
Simulation Environment for Surgical Robotics.

Provides:
- Physics simulation (PyBullet/MuJoCo backends)
- Surgical scene management
- Tissue deformation simulation
- Visualization tools
"""

from surgical_robotics.simulation.physics import (
    PhysicsEngine,
    PhysicsConfig,
    RigidBody,
    SoftBody,
)
from surgical_robotics.simulation.scene import (
    SurgicalScene,
    SceneObject,
    OperatingTable,
    PatientModel as SimPatientModel,
)
from surgical_robotics.simulation.tissue import (
    TissueSimulator,
    TissueProperties,
    DeformableTissue,
    CuttableTissue,
)
from surgical_robotics.simulation.robot_sim import (
    RobotSimulator,
    SimulatedJoint,
    SimulatedEndEffector,
)

__all__ = [
    "PhysicsEngine",
    "PhysicsConfig",
    "RigidBody",
    "SoftBody",
    "SurgicalScene",
    "SceneObject",
    "OperatingTable",
    "SimPatientModel",
    "TissueSimulator",
    "TissueProperties",
    "DeformableTissue",
    "CuttableTissue",
    "RobotSimulator",
    "SimulatedJoint",
    "SimulatedEndEffector",
]
