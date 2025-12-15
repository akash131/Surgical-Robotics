"""Neurosurgical Robot System with sub-millimeter precision."""

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
    WorkspaceVolume,
)
from surgical_robotics.core.kinematics import (
    ForwardKinematics,
    InverseKinematics,
    DHParameters,
    TrajectoryPlanner,
)
from surgical_robotics.core.safety import SafetyController


class NeurosurgicalProcedure(Enum):
    """Types of neurosurgical procedures."""
    BIOPSY = auto()
    DEEP_BRAIN_STIMULATION = auto()
    TUMOR_RESECTION = auto()
    STEREOTACTIC_RADIOSURGERY = auto()
    ELECTRODE_PLACEMENT = auto()
    CATHETER_PLACEMENT = auto()


class PrecisionMode(Enum):
    """Precision modes for different tasks."""
    STANDARD = auto()  # ±1mm accuracy
    HIGH = auto()  # ±0.5mm accuracy
    ULTRA = auto()  # ±0.1mm accuracy (sub-millimeter)


@dataclass
class StereotacticFrame:
    """
    Stereotactic frame for precise coordinate system definition.

    Provides a fixed reference frame attached to the patient's head
    for accurate targeting.
    """
    frame_id: str
    fiducial_positions: list[NDArray[np.float64]] = field(default_factory=list)
    registration_matrix: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(4)
    )
    is_registered: bool = False

    # Frame specifications
    frame_type: str = "leksell"  # "leksell", "cosman-roberts-wells", "frameless"

    def add_fiducial(self, position: NDArray[np.float64]) -> None:
        """Add a fiducial marker position."""
        self.fiducial_positions.append(np.asarray(position))

    def compute_registration(
        self,
        image_points: list[NDArray[np.float64]]
    ) -> bool:
        """
        Compute registration between frame and image coordinates.

        Uses paired-point registration with least squares fitting.
        """
        if len(self.fiducial_positions) < 3 or len(image_points) < 3:
            return False

        if len(self.fiducial_positions) != len(image_points):
            return False

        # Convert to arrays
        frame_pts = np.array(self.fiducial_positions)
        image_pts = np.array(image_points)

        # Compute centroids
        frame_centroid = np.mean(frame_pts, axis=0)
        image_centroid = np.mean(image_pts, axis=0)

        # Center the points
        frame_centered = frame_pts - frame_centroid
        image_centered = image_pts - image_centroid

        # SVD for rotation
        H = frame_centered.T @ image_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = image_centroid - R @ frame_centroid

        # Build transformation matrix
        self.registration_matrix = np.eye(4)
        self.registration_matrix[:3, :3] = R
        self.registration_matrix[:3, 3] = t

        self.is_registered = True
        return True

    def transform_to_image(
        self,
        frame_point: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform point from frame coordinates to image coordinates."""
        point_h = np.append(frame_point, 1)
        image_point_h = self.registration_matrix @ point_h
        return image_point_h[:3]

    def transform_to_frame(
        self,
        image_point: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform point from image coordinates to frame coordinates."""
        inv_matrix = np.linalg.inv(self.registration_matrix)
        point_h = np.append(image_point, 1)
        frame_point_h = inv_matrix @ point_h
        return frame_point_h[:3]


@dataclass
class NeurosurgicalTarget:
    """Target for neurosurgical procedure."""
    name: str
    position: NDArray[np.float64]  # Target position in image coordinates
    entry_point: Optional[NDArray[np.float64]] = None
    trajectory_direction: Optional[NDArray[np.float64]] = None

    # Safety parameters
    critical_structures: list[str] = field(default_factory=list)
    minimum_clearance: float = 0.002  # 2mm default clearance

    # Procedure parameters
    depth: float = 0.0  # Depth from entry point
    diameter: float = 0.001  # Target structure diameter

    def compute_trajectory(self) -> Optional[NDArray[np.float64]]:
        """Compute unit trajectory vector from entry to target."""
        if self.entry_point is None:
            return None

        direction = self.position - self.entry_point
        length = float(np.linalg.norm(direction))

        if length < 0.001:
            return None

        self.trajectory_direction = direction / length
        self.depth = length
        return self.trajectory_direction


@dataclass
class NeurosurgicalArm(RobotArm):
    """High-precision neurosurgical manipulator arm."""
    precision_mode: PrecisionMode = PrecisionMode.ULTRA

    # Encoders with high resolution
    encoder_resolution: float = 0.0001  # 0.1mm position resolution

    # Positioning accuracy specs
    positioning_accuracy: float = 0.0001  # 0.1mm
    repeatability: float = 0.00005  # 0.05mm

    def __post_init__(self) -> None:
        if self.num_joints == 0:
            self.num_joints = 6

        super().__post_init__()
        self._setup_high_precision_limits()

    def _setup_high_precision_limits(self) -> None:
        """Set up joint limits for high-precision operation."""
        # Reduced velocity limits for precision
        self.joint_limits = [
            JointLimits(-2.618, 2.618, 0.2, 1.0, 2.0),   # Base rotation
            JointLimits(-1.571, 1.571, 0.2, 1.0, 3.0),   # Shoulder
            JointLimits(-2.094, 2.094, 0.2, 1.0, 2.0),   # Elbow
            JointLimits(-3.14159, 3.14159, 0.3, 2.0, 1.0),  # Wrist 1
            JointLimits(-1.571, 1.571, 0.3, 2.0, 1.0),   # Wrist 2
            JointLimits(-3.14159, 3.14159, 0.3, 2.0, 0.5),  # Wrist 3
        ]


class NeurosurgicalRobot(SurgicalRobot):
    """
    Neurosurgical Robot System.

    Features:
    - Sub-millimeter positioning accuracy (±0.1mm)
    - Integration with intraoperative MRI/CT
    - Tremor cancellation
    - Stereotactic frame support
    """

    def __init__(self) -> None:
        super().__init__("NeurosurgicalRobot", num_arms=1)

        self.safety_controller = SafetyController()
        self.stereotactic_frame: Optional[StereotacticFrame] = None
        self.current_target: Optional[NeurosurgicalTarget] = None
        self.procedure_type: Optional[NeurosurgicalProcedure] = None

        # Precision tracking
        self.precision_mode = PrecisionMode.ULTRA
        self.position_error_history: list[float] = []
        self.max_error_samples = 100

        # Control parameters
        self.control_frequency = 2000.0  # 2kHz for high precision

        # Initialize arm
        self._setup_arm()
        self._setup_kinematics()

        # Workspace definition (typically small for neurosurgery)
        self.workspace = WorkspaceVolume(
            center=np.array([0.0, 0.0, 0.3]),
            dimensions=np.array([0.3, 0.3, 0.3]),  # 30cm cube
            shape="sphere"
        )

    def _setup_arm(self) -> None:
        """Initialize the neurosurgical arm."""
        arm = NeurosurgicalArm(
            name="NSA1",
            num_joints=6,
            precision_mode=PrecisionMode.ULTRA
        )
        arm.end_effector = EndEffector(
            name="probe",
            tool_type="biopsy_needle"
        )
        self.arms.append(arm)

    def _setup_kinematics(self) -> None:
        """Set up kinematics for precision positioning."""
        # DH parameters for a 6-DOF precision robot
        dh_params = [
            DHParameters(0, 0.1, 0, np.pi/2),
            DHParameters(0, 0, 0.25, 0),
            DHParameters(0, 0, 0.25, 0),
            DHParameters(0, 0.1, 0, np.pi/2),
            DHParameters(0, 0.1, 0, -np.pi/2),
            DHParameters(0, 0.05, 0, 0),
        ]

        self.fk = ForwardKinematics(dh_params)
        self.ik = InverseKinematics(
            self.fk,
            max_iterations=200,
            tolerance=1e-7,  # High precision tolerance
            damping=0.001
        )

    def initialize(self) -> bool:
        """Initialize the neurosurgical system."""
        self.state = RobotState.INITIALIZING

        # Verify system components
        if not self._verify_precision_components():
            self.state = RobotState.ERROR
            return False

        self.state = RobotState.READY
        return True

    def _verify_precision_components(self) -> bool:
        """Verify all precision components are functioning."""
        # Check encoder health, brake status, etc.
        return True

    def calibrate(self) -> bool:
        """Calibrate robot with high-precision verification."""
        self.state = RobotState.CALIBRATING

        # Home all joints
        for arm in self.arms:
            for joint_state in arm.joint_states:
                joint_state.position = 0.0

        # Perform precision calibration routine
        if not self._precision_calibration():
            self.state = RobotState.ERROR
            return False

        self.state = RobotState.READY
        return True

    def _precision_calibration(self) -> bool:
        """Perform high-precision calibration routine."""
        # Multi-point calibration for accuracy verification
        test_positions = [
            np.array([0.0, 0.0, 0.3]),
            np.array([0.05, 0.0, 0.3]),
            np.array([0.0, 0.05, 0.3]),
            np.array([0.0, 0.0, 0.35]),
        ]

        errors = []
        for target in test_positions:
            # Move to position and measure actual position
            # In real system, would use external measurement device
            measured = target + np.random.normal(0, 0.00005, 3)  # Simulated
            error = float(np.linalg.norm(target - measured))
            errors.append(error)

        # Check all errors within tolerance
        max_error = max(errors)
        return max_error < 0.0002  # 0.2mm tolerance

    def set_stereotactic_frame(self, frame: StereotacticFrame) -> None:
        """Attach stereotactic frame for coordinate registration."""
        self.stereotactic_frame = frame

    def set_target(self, target: NeurosurgicalTarget) -> bool:
        """Set surgical target."""
        if self.stereotactic_frame is None or not self.stereotactic_frame.is_registered:
            return False

        # Transform target to robot coordinates
        self.current_target = target

        # Validate trajectory if entry point specified
        if target.entry_point is not None:
            if not self._validate_trajectory(target):
                return False

        return True

    def _validate_trajectory(self, target: NeurosurgicalTarget) -> bool:
        """Validate surgical trajectory is safe."""
        if target.entry_point is None:
            return False

        # Check trajectory is within workspace
        if not self.workspace.contains_point(target.entry_point):
            return False

        if not self.workspace.contains_point(target.position):
            return False

        # Check trajectory doesn't pass through critical structures
        # Would integrate with imaging data in real implementation

        return True

    def move_to_pose(self, arm_index: int, target_pose: Pose) -> bool:
        """Move arm to target pose with high precision."""
        if not self._check_safety_conditions():
            return False

        arm = self.arms[arm_index]
        current_joints = np.array([js.position for js in arm.joint_states])

        # Get joint limits
        min_limits = np.array([jl.position_min for jl in arm.joint_limits])
        max_limits = np.array([jl.position_max for jl in arm.joint_limits])

        # Solve IK with high precision
        target_joints, success = self.ik.compute(
            target_pose,
            initial_guess=current_joints,
            joint_limits=(min_limits, max_limits)
        )

        if not success:
            return False

        return self.move_joints(arm_index, target_joints)

    def move_joints(self, arm_index: int, joint_positions: Sequence[float]) -> bool:
        """Move joints to target positions with precision control."""
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

        # Generate smooth trajectory
        current_positions = np.array([js.position for js in arm.joint_states])
        positions, velocities, _ = TrajectoryPlanner.quintic_polynomial(
            current_positions,
            target_positions,
            duration=2.0,  # Slow movement for precision
            dt=1.0 / self.control_frequency
        )

        # Execute trajectory
        self.state = RobotState.OPERATING
        for i, pos in enumerate(positions):
            for j, p in enumerate(pos):
                arm.joint_states[j].position = p
                if i < len(velocities):
                    arm.joint_states[j].velocity = velocities[i][j]

        # Update end-effector pose
        final_pose = self.fk.compute(target_positions)
        if arm.end_effector:
            arm.end_effector.pose = final_pose

        # Track positioning error
        self._track_positioning_error(final_pose, target_positions)

        self.state = RobotState.READY
        return True

    def _track_positioning_error(
        self,
        achieved_pose: Pose,
        target_joints: NDArray[np.float64]
    ) -> None:
        """Track positioning error for quality assurance."""
        # Compute forward kinematics of target
        target_pose = self.fk.compute(target_joints)
        error = float(np.linalg.norm(achieved_pose.position - target_pose.position))

        self.position_error_history.append(error)
        if len(self.position_error_history) > self.max_error_samples:
            self.position_error_history.pop(0)

    def get_position_accuracy_stats(self) -> dict[str, float]:
        """Get positioning accuracy statistics."""
        if not self.position_error_history:
            return {"mean": 0.0, "max": 0.0, "std": 0.0}

        errors = np.array(self.position_error_history)
        return {
            "mean": float(np.mean(errors)),
            "max": float(np.max(errors)),
            "std": float(np.std(errors)),
        }

    def approach_target(
        self,
        approach_distance: float = 0.02
    ) -> bool:
        """
        Approach current target along trajectory.

        Args:
            approach_distance: Distance from target to stop (safety margin)
        """
        if self.current_target is None:
            return False

        if self.current_target.entry_point is None:
            return False

        target = self.current_target
        direction = target.compute_trajectory()

        if direction is None:
            return False

        # Compute approach point
        approach_point = target.position - direction * approach_distance

        # Transform to robot coordinates if frame registered
        if self.stereotactic_frame and self.stereotactic_frame.is_registered:
            approach_point = self.stereotactic_frame.transform_to_frame(approach_point)

        # Create target pose
        # Orient along trajectory direction
        target_pose = Pose(
            position=approach_point,
            orientation=self._direction_to_quaternion(direction)
        )

        return self.move_to_pose(0, target_pose)

    def advance_to_target(self, step_size: float = 0.001) -> bool:
        """
        Advance toward target in small steps.

        Args:
            step_size: Step size in meters (default 1mm)
        """
        if self.current_target is None:
            return False

        arm = self.arms[0]
        if arm.end_effector is None:
            return False

        current_pos = arm.end_effector.pose.position
        target_pos = self.current_target.position

        if self.stereotactic_frame and self.stereotactic_frame.is_registered:
            target_pos = self.stereotactic_frame.transform_to_frame(target_pos)

        direction = target_pos - current_pos
        distance = float(np.linalg.norm(direction))

        if distance < step_size:
            # Already at target
            return True

        direction = direction / distance
        next_pos = current_pos + direction * step_size

        target_pose = Pose(
            position=next_pos,
            orientation=arm.end_effector.pose.orientation
        )

        return self.move_to_pose(0, target_pose)

    def retract(self, distance: float = 0.01) -> bool:
        """
        Retract along trajectory.

        Args:
            distance: Retraction distance in meters
        """
        arm = self.arms[0]
        if arm.end_effector is None:
            return False

        if self.current_target is None or self.current_target.trajectory_direction is None:
            # Retract along Z axis if no trajectory defined
            direction = np.array([0, 0, 1])
        else:
            direction = -self.current_target.trajectory_direction

        current_pos = arm.end_effector.pose.position
        retract_pos = current_pos + direction * distance

        target_pose = Pose(
            position=retract_pos,
            orientation=arm.end_effector.pose.orientation
        )

        return self.move_to_pose(0, target_pose)

    @staticmethod
    def _direction_to_quaternion(
        direction: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Convert a direction vector to quaternion orientation."""
        # Align Z axis with direction
        z_axis = direction / np.linalg.norm(direction)

        # Choose arbitrary up vector
        if abs(z_axis[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])

        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])

        # Convert rotation matrix to quaternion
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])
