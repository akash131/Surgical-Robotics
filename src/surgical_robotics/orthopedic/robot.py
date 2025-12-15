"""Orthopedic Robot System for bone surgery."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import (
    SurgicalRobot,
    RobotArm,
    RobotState,
    Pose,
    JointState,
    JointLimits,
    EndEffector,
)
from surgical_robotics.core.kinematics import (
    ForwardKinematics,
    InverseKinematics,
    DHParameters,
    TrajectoryPlanner,
)
from surgical_robotics.core.safety import SafetyController


class CuttingMode(Enum):
    """Bone cutting operational modes."""
    IDLE = auto()
    SAW = auto()
    BURR = auto()
    DRILL = auto()
    REAMING = auto()


class OrthopedicProcedure(Enum):
    """Types of orthopedic procedures."""
    TOTAL_KNEE_ARTHROPLASTY = auto()
    TOTAL_HIP_ARTHROPLASTY = auto()
    UNICONDYLAR_KNEE = auto()
    SPINE_FUSION = auto()
    PEDICLE_SCREW = auto()
    OSTEOTOMY = auto()


@dataclass
class BoneRegistration:
    """Registration of bone anatomy to robot coordinates."""
    bone_name: str
    landmark_points: list[NDArray[np.float64]] = field(default_factory=list)
    surface_points: list[NDArray[np.float64]] = field(default_factory=list)
    registration_matrix: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(4)
    )
    registration_error: float = float('inf')
    is_registered: bool = False

    def add_landmark(self, point: NDArray[np.float64]) -> None:
        """Add anatomical landmark point."""
        self.landmark_points.append(np.asarray(point))

    def add_surface_point(self, point: NDArray[np.float64]) -> None:
        """Add surface digitization point."""
        self.surface_points.append(np.asarray(point))


@dataclass
class OrthopedicArm(RobotArm):
    """Orthopedic surgical robot arm."""
    cutting_mode: CuttingMode = CuttingMode.IDLE

    # Tool parameters
    spindle_speed: float = 0.0  # RPM
    feed_rate: float = 0.0  # mm/s

    # Force sensing
    measured_force: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    measured_torque: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )

    def __post_init__(self) -> None:
        if self.num_joints == 0:
            self.num_joints = 6

        super().__post_init__()
        self._setup_joint_limits()

    def _setup_joint_limits(self) -> None:
        """Set up joint limits for orthopedic operations."""
        # Robust limits for bone cutting forces
        self.joint_limits = [
            JointLimits(-3.14159, 3.14159, 1.0, 5.0, 100.0),   # Base rotation
            JointLimits(-2.094, 2.094, 1.0, 5.0, 150.0),       # Shoulder
            JointLimits(-2.618, 2.618, 1.0, 5.0, 100.0),       # Elbow
            JointLimits(-3.14159, 3.14159, 2.0, 10.0, 50.0),   # Wrist 1
            JointLimits(-2.094, 2.094, 2.0, 10.0, 50.0),       # Wrist 2
            JointLimits(-3.14159, 3.14159, 2.0, 10.0, 30.0),   # Wrist 3
        ]


@dataclass
class CuttingBoundary:
    """Defines boundaries for safe cutting."""
    name: str
    boundary_points: list[NDArray[np.float64]]
    normal_direction: NDArray[np.float64]
    max_depth: float  # Maximum cutting depth in meters
    is_active: bool = True

    def is_within_boundary(self, point: NDArray[np.float64]) -> bool:
        """Check if point is within cutting boundary."""
        if len(self.boundary_points) < 3:
            return False

        # Project point onto plane
        plane_point = self.boundary_points[0]
        normal = self.normal_direction / np.linalg.norm(self.normal_direction)

        # Distance from plane
        dist = np.dot(point - plane_point, normal)
        if dist > self.max_depth:
            return False

        # Check if within polygon (simplified 2D check)
        # Project to plane perpendicular to normal
        return True  # Simplified


class OrthopedicRobot(SurgicalRobot):
    """
    Orthopedic Robot System.

    Features:
    - Bone cutting and milling systems
    - Force feedback for tissue differentiation
    - Patient-specific surgical planning
    - Haptic boundaries for safe cutting
    """

    def __init__(self) -> None:
        super().__init__("OrthopedicRobot", num_arms=1)

        self.safety_controller = SafetyController()
        self.procedure_type: Optional[OrthopedicProcedure] = None
        self.cutting_mode = CuttingMode.IDLE

        # Bone registration
        self.bone_registrations: dict[str, BoneRegistration] = {}

        # Cutting boundaries (haptic walls)
        self.cutting_boundaries: list[CuttingBoundary] = []

        # Force control parameters
        self.max_cutting_force = 50.0  # Newtons
        self.force_control_enabled = True
        self.force_setpoint = 10.0  # Target cutting force

        # Tool state
        self.spindle_running = False
        self.spindle_speed = 0.0  # RPM
        self.coolant_active = False

        # Initialize arm
        self._setup_arm()
        self._setup_kinematics()

    def _setup_arm(self) -> None:
        """Initialize the orthopedic arm."""
        arm = OrthopedicArm(
            name="OrthoArm1",
            num_joints=6
        )
        arm.end_effector = EndEffector(
            name="cutting_tool",
            tool_type="burr"
        )
        self.arms.append(arm)

    def _setup_kinematics(self) -> None:
        """Set up kinematics for bone cutting operations."""
        # DH parameters for 6-DOF orthopedic robot
        dh_params = [
            DHParameters(0, 0.15, 0, np.pi/2),
            DHParameters(0, 0, 0.35, 0),
            DHParameters(0, 0, 0.30, 0),
            DHParameters(0, 0.12, 0, np.pi/2),
            DHParameters(0, 0.10, 0, -np.pi/2),
            DHParameters(0, 0.08, 0, 0),
        ]

        self.fk = ForwardKinematics(dh_params)
        self.ik = InverseKinematics(
            self.fk,
            max_iterations=100,
            tolerance=1e-5,
            damping=0.05
        )

    def initialize(self) -> bool:
        """Initialize the orthopedic system."""
        self.state = RobotState.INITIALIZING

        # System checks
        if not self._verify_tool_mounted():
            self.state = RobotState.ERROR
            return False

        if not self._verify_force_sensors():
            self.state = RobotState.ERROR
            return False

        self.state = RobotState.READY
        return True

    def _verify_tool_mounted(self) -> bool:
        """Verify cutting tool is properly mounted."""
        arm = self.arms[0]
        return arm.end_effector is not None

    def _verify_force_sensors(self) -> bool:
        """Verify force/torque sensors are functioning."""
        # Would check actual sensor readings
        return True

    def calibrate(self) -> bool:
        """Calibrate robot and tools."""
        self.state = RobotState.CALIBRATING

        # Home all joints
        for arm in self.arms:
            for joint_state in arm.joint_states:
                joint_state.position = 0.0

        # Calibrate tool center point
        if not self._calibrate_tcp():
            self.state = RobotState.ERROR
            return False

        self.state = RobotState.READY
        return True

    def _calibrate_tcp(self) -> bool:
        """Calibrate tool center point."""
        # Would perform 4-point TCP calibration
        return True

    def register_bone(
        self,
        bone_name: str,
        ct_landmarks: list[NDArray[np.float64]],
        robot_landmarks: list[NDArray[np.float64]]
    ) -> bool:
        """
        Register bone anatomy to robot coordinates.

        Args:
            bone_name: Name of bone (e.g., "femur", "tibia")
            ct_landmarks: Landmark points from CT scan
            robot_landmarks: Corresponding points digitized by robot

        Returns:
            True if registration successful
        """
        if len(ct_landmarks) < 3 or len(robot_landmarks) < 3:
            return False

        if len(ct_landmarks) != len(robot_landmarks):
            return False

        registration = BoneRegistration(bone_name=bone_name)

        # Compute registration using paired-point method
        ct_pts = np.array(ct_landmarks)
        robot_pts = np.array(robot_landmarks)

        # Centroids
        ct_centroid = np.mean(ct_pts, axis=0)
        robot_centroid = np.mean(robot_pts, axis=0)

        # Center points
        ct_centered = ct_pts - ct_centroid
        robot_centered = robot_pts - robot_centroid

        # SVD for rotation
        H = ct_centered.T @ robot_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = robot_centroid - R @ ct_centroid

        # Build transformation
        registration.registration_matrix[:3, :3] = R
        registration.registration_matrix[:3, 3] = t

        # Compute error
        transformed = (R @ ct_pts.T).T + t
        errors = np.linalg.norm(transformed - robot_pts, axis=1)
        registration.registration_error = float(np.sqrt(np.mean(errors ** 2)))

        # Accept if error is acceptable
        if registration.registration_error < 0.002:  # 2mm threshold
            registration.is_registered = True
            self.bone_registrations[bone_name] = registration
            return True

        return False

    def add_cutting_boundary(self, boundary: CuttingBoundary) -> None:
        """Add a cutting boundary for haptic guidance."""
        self.cutting_boundaries.append(boundary)

    def clear_cutting_boundaries(self) -> None:
        """Clear all cutting boundaries."""
        self.cutting_boundaries = []

    def set_cutting_mode(self, mode: CuttingMode) -> bool:
        """Set cutting mode and configure tool."""
        if self.state not in [RobotState.READY, RobotState.OPERATING]:
            return False

        self.cutting_mode = mode
        arm = self.arms[0]

        if isinstance(arm, OrthopedicArm):
            arm.cutting_mode = mode

            # Configure spindle speed based on mode
            if mode == CuttingMode.SAW:
                self.spindle_speed = 12000  # RPM
            elif mode == CuttingMode.BURR:
                self.spindle_speed = 80000  # RPM
            elif mode == CuttingMode.DRILL:
                self.spindle_speed = 1500  # RPM
            elif mode == CuttingMode.REAMING:
                self.spindle_speed = 200  # RPM
            else:
                self.spindle_speed = 0

        return True

    def start_spindle(self) -> bool:
        """Start cutting tool spindle."""
        if self.cutting_mode == CuttingMode.IDLE:
            return False

        self.spindle_running = True
        return True

    def stop_spindle(self) -> None:
        """Stop cutting tool spindle."""
        self.spindle_running = False
        self.spindle_speed = 0

    def move_to_pose(self, arm_index: int, target_pose: Pose) -> bool:
        """Move arm to target pose with boundary checking."""
        if not self._check_safety_conditions():
            return False

        # Check cutting boundaries
        if self.cutting_boundaries:
            boundary_check = self._check_boundary_constraints(target_pose.position)
            if not boundary_check["within_all"]:
                return False

        arm = self.arms[arm_index]
        current_joints = np.array([js.position for js in arm.joint_states])

        min_limits = np.array([jl.position_min for jl in arm.joint_limits])
        max_limits = np.array([jl.position_max for jl in arm.joint_limits])

        target_joints, success = self.ik.compute(
            target_pose,
            initial_guess=current_joints,
            joint_limits=(min_limits, max_limits)
        )

        if not success:
            return False

        return self.move_joints(arm_index, target_joints)

    def move_joints(self, arm_index: int, joint_positions: Sequence[float]) -> bool:
        """Move joints with force monitoring."""
        if not self._check_safety_conditions():
            return False

        if arm_index >= len(self.arms):
            return False

        arm = self.arms[arm_index]
        target_positions = np.asarray(joint_positions)

        # Validate limits
        for i, (pos, limits) in enumerate(zip(target_positions, arm.joint_limits)):
            if pos < limits.position_min or pos > limits.position_max:
                return False

        # Generate trajectory
        current_positions = np.array([js.position for js in arm.joint_states])
        positions, velocities, _ = TrajectoryPlanner.quintic_polynomial(
            current_positions,
            target_positions,
            duration=1.0,
            dt=0.001
        )

        # Execute with force monitoring
        self.state = RobotState.OPERATING
        for i, pos in enumerate(positions):
            for j, p in enumerate(pos):
                arm.joint_states[j].position = p

            # Check force limits during cutting
            if self.spindle_running and isinstance(arm, OrthopedicArm):
                force_magnitude = float(np.linalg.norm(arm.measured_force))
                if force_magnitude > self.max_cutting_force:
                    self.stop_spindle()
                    self.state = RobotState.PAUSED
                    return False

        # Update end-effector pose
        final_pose = self.fk.compute(target_positions)
        if arm.end_effector:
            arm.end_effector.pose = final_pose

        self.state = RobotState.READY
        return True

    def _check_boundary_constraints(
        self,
        position: NDArray[np.float64]
    ) -> dict[str, any]:
        """Check position against all cutting boundaries."""
        result = {
            "within_all": True,
            "violated_boundaries": [],
            "distances": {}
        }

        for boundary in self.cutting_boundaries:
            if not boundary.is_active:
                continue

            within = boundary.is_within_boundary(position)
            if not within:
                result["within_all"] = False
                result["violated_boundaries"].append(boundary.name)

        return result

    def execute_cutting_path(
        self,
        path_points: list[NDArray[np.float64]],
        feed_rate: float = 0.005  # 5mm/s default
    ) -> bool:
        """
        Execute a cutting path.

        Args:
            path_points: List of waypoints for cutting
            feed_rate: Feed rate in m/s

        Returns:
            True if path completed successfully
        """
        if not self.spindle_running:
            return False

        if not path_points:
            return False

        arm = self.arms[0]
        if isinstance(arm, OrthopedicArm):
            arm.feed_rate = feed_rate * 1000  # Convert to mm/s

        for target_point in path_points:
            # Check boundary constraints
            if self.cutting_boundaries:
                boundary_check = self._check_boundary_constraints(target_point)
                if not boundary_check["within_all"]:
                    self.stop_spindle()
                    return False

            # Create target pose (maintain current orientation)
            current_pose = self.get_arm_pose(0)
            if current_pose is None:
                return False

            target_pose = Pose(
                position=target_point,
                orientation=current_pose.orientation
            )

            if not self.move_to_pose(0, target_pose):
                self.stop_spindle()
                return False

        return True

    def update_force_reading(
        self,
        force: NDArray[np.float64],
        torque: NDArray[np.float64]
    ) -> None:
        """Update force/torque sensor readings."""
        arm = self.arms[0]
        if isinstance(arm, OrthopedicArm):
            arm.measured_force = np.asarray(force)
            arm.measured_torque = np.asarray(torque)

            # Check force limits
            force_magnitude = float(np.linalg.norm(force))
            if force_magnitude > self.max_cutting_force:
                self.stop_spindle()
                self.state = RobotState.PAUSED

    def get_cutting_force(self) -> float:
        """Get current cutting force magnitude."""
        arm = self.arms[0]
        if isinstance(arm, OrthopedicArm):
            return float(np.linalg.norm(arm.measured_force))
        return 0.0

    def set_coolant(self, active: bool) -> None:
        """Enable/disable coolant for cutting."""
        self.coolant_active = active

    def transform_to_robot(
        self,
        bone_name: str,
        ct_point: NDArray[np.float64]
    ) -> Optional[NDArray[np.float64]]:
        """Transform point from CT to robot coordinates."""
        if bone_name not in self.bone_registrations:
            return None

        registration = self.bone_registrations[bone_name]
        if not registration.is_registered:
            return None

        point_h = np.append(ct_point, 1)
        robot_point_h = registration.registration_matrix @ point_h
        return robot_point_h[:3]
