"""Da Vinci-Class Multi-Arm Surgical Robot System."""

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
from surgical_robotics.core.safety import SafetyController, CollisionDetector


class ArmRole(Enum):
    """Role of each arm in the Da Vinci system."""
    PATIENT_SIDE_LEFT = auto()
    PATIENT_SIDE_RIGHT = auto()
    PATIENT_SIDE_AUX = auto()  # Third arm
    ENDOSCOPE = auto()  # Camera arm


@dataclass
class DaVinciArm(RobotArm):
    """Da Vinci patient-side manipulator arm."""
    role: ArmRole = ArmRole.PATIENT_SIDE_LEFT
    remote_center_of_motion: Optional[NDArray[np.float64]] = None  # RCM point
    cannula_position: Optional[NDArray[np.float64]] = None

    # Typical Da Vinci has 7 DOF per arm
    # 3 DOF for positioning, 1 DOF insertion, 3 DOF wrist
    def __post_init__(self) -> None:
        if self.num_joints == 0:
            self.num_joints = 7
        super().__post_init__()
        self._setup_joint_limits()

    def _setup_joint_limits(self) -> None:
        """Set up typical joint limits for Da Vinci arm."""
        # Joint limits based on typical surgical workspace
        self.joint_limits = [
            JointLimits(-1.5708, 1.5708, 1.0, 5.0, 5.0),   # Outer yaw
            JointLimits(-0.7854, 0.7854, 1.0, 5.0, 5.0),   # Outer pitch
            JointLimits(0.0, 0.24, 0.05, 0.5, 10.0),       # Insertion (prismatic)
            JointLimits(-2.618, 2.618, 2.0, 10.0, 2.0),    # Outer roll
            JointLimits(-1.309, 1.309, 2.0, 10.0, 1.0),    # Wrist pitch
            JointLimits(-1.309, 1.309, 2.0, 10.0, 1.0),    # Wrist yaw
            JointLimits(-3.14159, 3.14159, 3.0, 15.0, 0.5), # Gripper roll
        ]


class MasterConsole:
    """Surgeon's master console with hand controllers."""

    def __init__(self) -> None:
        self.left_controller_pose = Pose.identity()
        self.right_controller_pose = Pose.identity()
        self.clutch_engaged = False
        self.camera_control_mode = False
        self.scaling_factor = 0.5  # Motion scaling (1:2 default)

        # Foot pedal states
        self.clutch_pedal = False
        self.camera_pedal = False
        self.coag_pedal = False
        self.bipolar_pedal = False

    def get_scaled_motion(
        self, controller_pose: Pose, previous_pose: Pose
    ) -> NDArray[np.float64]:
        """Get scaled motion delta for robot control."""
        if self.clutch_engaged:
            return np.zeros(3)

        delta = controller_pose.position - previous_pose.position
        return delta * self.scaling_factor

    def set_scaling(self, scaling: float) -> None:
        """Set motion scaling factor (0.2 to 1.0)."""
        self.scaling_factor = np.clip(scaling, 0.2, 1.0)


class DaVinciRobot(SurgicalRobot):
    """
    Da Vinci-Class Multi-Arm Surgical Robot System.

    Features:
    - Multiple patient-side arms (typically 3-4)
    - Remote center of motion (RCM) constraint
    - Master-slave teleoperation
    - Haptic feedback support
    - 3D stereoscopic vision
    """

    def __init__(self, num_patient_arms: int = 3, has_endoscope: bool = True) -> None:
        super().__init__("DaVinci", num_arms=num_patient_arms + (1 if has_endoscope else 0))
        self.num_patient_arms = num_patient_arms
        self.has_endoscope = has_endoscope

        self.master_console = MasterConsole()
        self.safety_controller = SafetyController()
        self.collision_detector = CollisionDetector()

        # Initialize arms
        self._setup_arms()

        # Kinematics
        self._setup_kinematics()

        # Teleoperation state
        self.teleoperation_active = False
        self.arm_to_controller_mapping: dict[int, str] = {
            0: "left",
            1: "right",
        }

    def _setup_arms(self) -> None:
        """Initialize all robot arms."""
        arm_roles = [
            ArmRole.PATIENT_SIDE_LEFT,
            ArmRole.PATIENT_SIDE_RIGHT,
            ArmRole.PATIENT_SIDE_AUX,
        ]

        for i in range(self.num_patient_arms):
            role = arm_roles[i] if i < len(arm_roles) else ArmRole.PATIENT_SIDE_AUX
            arm = DaVinciArm(
                name=f"PSM{i+1}",
                num_joints=7,
                role=role,
            )
            arm.end_effector = EndEffector(
                name=f"Instrument_{i+1}",
                tool_type="grasper"
            )
            self.arms.append(arm)

        if self.has_endoscope:
            endoscope_arm = DaVinciArm(
                name="ECM",
                num_joints=4,  # Endoscope has fewer DOF
                role=ArmRole.ENDOSCOPE,
            )
            endoscope_arm.end_effector = EndEffector(
                name="Endoscope",
                tool_type="camera"
            )
            self.arms.append(endoscope_arm)

    def _setup_kinematics(self) -> None:
        """Set up forward and inverse kinematics."""
        # Typical Da Vinci PSM DH parameters (simplified)
        psm_dh_params = [
            DHParameters(0, 0, 0, np.pi/2),      # Joint 1
            DHParameters(0, 0, 0, -np.pi/2),     # Joint 2
            DHParameters(0, 0, 0, np.pi/2),      # Joint 3 (insertion)
            DHParameters(0, 0.4318, 0, 0),       # Joint 4
            DHParameters(0, 0, 0, -np.pi/2),     # Joint 5
            DHParameters(0, 0.4318, 0, np.pi/2), # Joint 6
            DHParameters(0, 0, 0, 0),            # Joint 7
        ]

        self.fk_solvers: dict[int, ForwardKinematics] = {}
        self.ik_solvers: dict[int, InverseKinematics] = {}

        for i in range(self.num_patient_arms):
            fk = ForwardKinematics(psm_dh_params)
            self.fk_solvers[i] = fk
            self.ik_solvers[i] = InverseKinematics(fk)

    def initialize(self) -> bool:
        """Initialize the Da Vinci system."""
        self.state = RobotState.INITIALIZING

        # Perform system checks
        if not self._check_communication():
            self.state = RobotState.ERROR
            return False

        if not self._check_instruments():
            self.state = RobotState.ERROR
            return False

        self.state = RobotState.READY
        return True

    def _check_communication(self) -> bool:
        """Check communication with all subsystems."""
        # Simulated communication check
        return True

    def _check_instruments(self) -> bool:
        """Verify instruments are properly installed."""
        for arm in self.arms:
            if arm.end_effector is None:
                return False
        return True

    def calibrate(self) -> bool:
        """Calibrate robot kinematics and sensors."""
        self.state = RobotState.CALIBRATING

        # Home all joints
        for arm in self.arms:
            for joint_state in arm.joint_states:
                joint_state.position = 0.0
                joint_state.velocity = 0.0

        self.state = RobotState.READY
        return True

    def move_to_pose(self, arm_index: int, target_pose: Pose) -> bool:
        """Move specified arm to target pose using IK."""
        if not self._check_safety_conditions():
            return False

        if arm_index not in self.ik_solvers:
            return False

        arm = self.arms[arm_index]
        current_joints = np.array([js.position for js in arm.joint_states])

        # Get joint limits
        min_limits = np.array([jl.position_min for jl in arm.joint_limits])
        max_limits = np.array([jl.position_max for jl in arm.joint_limits])

        # Solve IK
        target_joints, success = self.ik_solvers[arm_index].compute(
            target_pose,
            initial_guess=current_joints,
            joint_limits=(min_limits, max_limits)
        )

        if not success:
            return False

        # Check for collisions along path
        collision = self.collision_detector.check_path_collision(
            current_joints[:3], target_joints[:3]
        )
        if collision.collision_detected:
            return False

        return self.move_joints(arm_index, target_joints)

    def move_joints(self, arm_index: int, joint_positions: Sequence[float]) -> bool:
        """Move specified arm joints to target positions."""
        if not self._check_safety_conditions():
            return False

        if arm_index >= len(self.arms):
            return False

        arm = self.arms[arm_index]
        joint_positions_arr = np.asarray(joint_positions)

        # Validate joint limits
        for i, (pos, limits) in enumerate(zip(joint_positions_arr, arm.joint_limits)):
            if pos < limits.position_min or pos > limits.position_max:
                return False

        # Apply RCM constraint if set
        if isinstance(arm, DaVinciArm) and arm.remote_center_of_motion is not None:
            if not self._validate_rcm_constraint(arm_index, joint_positions_arr):
                return False

        # Update joint states
        self.state = RobotState.OPERATING
        for i, pos in enumerate(joint_positions_arr):
            if i < len(arm.joint_states):
                arm.joint_states[i].position = pos

        # Update end-effector pose
        if arm_index in self.fk_solvers:
            arm.end_effector.pose = self.fk_solvers[arm_index].compute(joint_positions_arr)

        self.state = RobotState.READY
        return True

    def _validate_rcm_constraint(
        self, arm_index: int, joint_positions: NDArray[np.float64]
    ) -> bool:
        """Validate that motion respects remote center of motion constraint."""
        arm = self.arms[arm_index]
        if not isinstance(arm, DaVinciArm) or arm.remote_center_of_motion is None:
            return True

        # Check that instrument passes through RCM point
        # Simplified check - in reality would verify full kinematic chain
        rcm = arm.remote_center_of_motion
        if arm_index in self.fk_solvers:
            frames = self.fk_solvers[arm_index].compute_all_frames(joint_positions)
            if len(frames) > 3:
                insertion_point = frames[3][:3, 3]
                distance_to_rcm = np.linalg.norm(insertion_point - rcm)
                return distance_to_rcm < 0.005  # 5mm tolerance

        return True

    def set_rcm(self, arm_index: int, rcm_position: NDArray[np.float64]) -> None:
        """Set remote center of motion for an arm."""
        if arm_index < len(self.arms):
            arm = self.arms[arm_index]
            if isinstance(arm, DaVinciArm):
                arm.remote_center_of_motion = np.asarray(rcm_position)

    def start_teleoperation(self) -> bool:
        """Start master-slave teleoperation mode."""
        if self.state != RobotState.READY:
            return False

        self.teleoperation_active = True
        self.state = RobotState.OPERATING
        return True

    def stop_teleoperation(self) -> None:
        """Stop teleoperation mode."""
        self.teleoperation_active = False
        self.state = RobotState.READY

    def update_teleoperation(self) -> None:
        """
        Update arm positions based on master console input.
        Called at control loop frequency.
        """
        if not self.teleoperation_active:
            return

        if self.master_console.clutch_engaged:
            return

        # Update mapped arms
        for arm_idx, controller in self.arm_to_controller_mapping.items():
            if arm_idx >= len(self.arms):
                continue

            controller_pose = (
                self.master_console.left_controller_pose
                if controller == "left"
                else self.master_console.right_controller_pose
            )

            # Apply scaled motion to arm
            arm = self.arms[arm_idx]
            if arm.end_effector:
                current_pose = arm.end_effector.pose
                motion_delta = self.master_console.get_scaled_motion(
                    controller_pose, current_pose
                )
                new_position = current_pose.position + motion_delta

                target_pose = Pose(
                    position=new_position,
                    orientation=controller_pose.orientation
                )
                self.move_to_pose(arm_idx, target_pose)

    def swap_arms(self, arm1_index: int, arm2_index: int) -> bool:
        """Swap controller mappings between two arms."""
        if arm1_index in self.arm_to_controller_mapping:
            controller1 = self.arm_to_controller_mapping.get(arm1_index)
            controller2 = self.arm_to_controller_mapping.get(arm2_index)

            if controller1 and controller2:
                self.arm_to_controller_mapping[arm1_index] = controller2
                self.arm_to_controller_mapping[arm2_index] = controller1
                return True
        return False

    def engage_instrument(self, arm_index: int, instrument_type: str) -> bool:
        """Engage/activate instrument on specified arm."""
        if arm_index >= len(self.arms):
            return False

        arm = self.arms[arm_index]
        if arm.end_effector:
            arm.end_effector.is_active = True
            arm.end_effector.tool_type = instrument_type
            return True
        return False

    def set_grip_force(self, arm_index: int, force: float) -> bool:
        """Set gripper force for grasping instruments."""
        if arm_index >= len(self.arms):
            return False

        arm = self.arms[arm_index]
        if arm.end_effector and arm.end_effector.tool_type in ["grasper", "needle_driver"]:
            # Limit grip force to safe range
            arm.end_effector.grip_force = np.clip(force, 0, 10.0)  # 0-10N
            return True
        return False

    def articulate_wrist(self, arm_index: int, pitch: float, yaw: float) -> bool:
        """Articulate wrist joints for fine manipulation."""
        if arm_index >= len(self.arms):
            return False

        arm = self.arms[arm_index]
        if len(arm.joint_states) >= 6:
            # Joints 5 and 6 are wrist pitch and yaw
            arm.joint_states[4].position = np.clip(pitch, -1.309, 1.309)
            arm.joint_states[5].position = np.clip(yaw, -1.309, 1.309)
            return True
        return False
