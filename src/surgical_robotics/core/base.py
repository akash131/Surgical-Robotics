"""Base classes for surgical robotics systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray


class RobotState(Enum):
    """Operational state of the surgical robot."""
    IDLE = auto()
    INITIALIZING = auto()
    CALIBRATING = auto()
    READY = auto()
    OPERATING = auto()
    PAUSED = auto()
    EMERGENCY_STOP = auto()
    ERROR = auto()


class SafetyLevel(Enum):
    """Safety classification levels."""
    NORMAL = auto()
    CAUTION = auto()
    WARNING = auto()
    CRITICAL = auto()


@dataclass
class Pose:
    """6-DOF pose representation (position + orientation)."""
    position: NDArray[np.float64]  # [x, y, z] in meters
    orientation: NDArray[np.float64]  # quaternion [w, x, y, z]

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        # Normalize quaternion
        self.orientation = self.orientation / np.linalg.norm(self.orientation)

    @classmethod
    def identity(cls) -> "Pose":
        """Create identity pose at origin."""
        return cls(
            position=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )

    def to_matrix(self) -> NDArray[np.float64]:
        """Convert to 4x4 homogeneous transformation matrix."""
        w, x, y, z = self.orientation
        rotation = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.position
        return matrix


@dataclass
class JointState:
    """State of a single robot joint."""
    position: float  # radians for revolute, meters for prismatic
    velocity: float = 0.0
    acceleration: float = 0.0
    torque: float = 0.0
    temperature: float = 25.0  # Celsius


@dataclass
class JointLimits:
    """Operating limits for a joint."""
    position_min: float
    position_max: float
    velocity_max: float
    acceleration_max: float
    torque_max: float


@dataclass
class EndEffector:
    """Surgical end effector (instrument tip)."""
    name: str
    tool_type: str  # e.g., "grasper", "scissors", "cautery", "drill"
    pose: Pose = field(default_factory=Pose.identity)
    is_active: bool = False
    grip_force: float = 0.0  # Newtons

    # Tool-specific parameters
    articulation_angle: float = 0.0  # degrees
    rotation_angle: float = 0.0  # degrees


@dataclass
class RobotArm:
    """Individual robot arm with multiple joints."""
    name: str
    num_joints: int
    joint_states: list[JointState] = field(default_factory=list)
    joint_limits: list[JointLimits] = field(default_factory=list)
    end_effector: Optional[EndEffector] = None
    base_pose: Pose = field(default_factory=Pose.identity)

    # DH parameters for kinematics
    dh_parameters: Optional[NDArray[np.float64]] = None  # [theta, d, a, alpha]

    def __post_init__(self) -> None:
        if not self.joint_states:
            self.joint_states = [JointState(position=0.0) for _ in range(self.num_joints)]


class SurgicalRobot(ABC):
    """Abstract base class for surgical robots."""

    def __init__(self, name: str, num_arms: int = 1) -> None:
        self.name = name
        self.num_arms = num_arms
        self.state = RobotState.IDLE
        self.safety_level = SafetyLevel.NORMAL
        self.arms: list[RobotArm] = []
        self.control_frequency: float = 1000.0  # Hz
        self._emergency_stop_active = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the robot system."""
        pass

    @abstractmethod
    def calibrate(self) -> bool:
        """Calibrate robot kinematics and sensors."""
        pass

    @abstractmethod
    def move_to_pose(self, arm_index: int, target_pose: Pose) -> bool:
        """Move specified arm to target pose."""
        pass

    @abstractmethod
    def move_joints(self, arm_index: int, joint_positions: Sequence[float]) -> bool:
        """Move specified arm joints to target positions."""
        pass

    def emergency_stop(self) -> None:
        """Trigger emergency stop - halt all motion immediately."""
        self._emergency_stop_active = True
        self.state = RobotState.EMERGENCY_STOP
        self.safety_level = SafetyLevel.CRITICAL

    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop if safe to do so."""
        if self._check_safety_conditions():
            self._emergency_stop_active = False
            self.state = RobotState.IDLE
            self.safety_level = SafetyLevel.NORMAL
            return True
        return False

    def _check_safety_conditions(self) -> bool:
        """Check if it's safe to operate."""
        return not self._emergency_stop_active and self.safety_level != SafetyLevel.CRITICAL

    def get_arm_pose(self, arm_index: int) -> Optional[Pose]:
        """Get current end-effector pose for specified arm."""
        if 0 <= arm_index < len(self.arms):
            arm = self.arms[arm_index]
            if arm.end_effector:
                return arm.end_effector.pose
        return None

    def get_joint_positions(self, arm_index: int) -> Optional[list[float]]:
        """Get current joint positions for specified arm."""
        if 0 <= arm_index < len(self.arms):
            return [js.position for js in self.arms[arm_index].joint_states]
        return None


@dataclass
class WorkspaceVolume:
    """Defines the allowable workspace for robot operation."""
    center: NDArray[np.float64]
    dimensions: NDArray[np.float64]  # [width, height, depth]
    shape: str = "cuboid"  # "cuboid", "sphere", "cylinder"

    def contains_point(self, point: NDArray[np.float64]) -> bool:
        """Check if a point is within the workspace."""
        if self.shape == "cuboid":
            half_dims = self.dimensions / 2
            return np.all(np.abs(point - self.center) <= half_dims)
        elif self.shape == "sphere":
            radius = self.dimensions[0] / 2
            return float(np.linalg.norm(point - self.center)) <= radius
        return False
