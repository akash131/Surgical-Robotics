"""Robot simulation for surgical robotics."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable, Tuple
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import Pose


class JointType(Enum):
    """Types of robot joints."""
    REVOLUTE = auto()  # Rotational
    PRISMATIC = auto()  # Linear
    FIXED = auto()  # No motion


@dataclass
class SimulatedJoint:
    """Simulated robot joint."""
    name: str
    joint_type: JointType

    # Kinematics
    axis: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0, 0, 1])
    )
    parent_link: Optional[str] = None
    child_link: Optional[str] = None

    # DH parameters
    d: float = 0.0  # Link offset
    a: float = 0.0  # Link length
    alpha: float = 0.0  # Link twist

    # Limits
    position_min: float = -np.pi
    position_max: float = np.pi
    velocity_max: float = 2.0  # rad/s or m/s
    torque_max: float = 100.0  # Nm or N

    # Current state
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0

    # Dynamics
    inertia: float = 0.1
    damping: float = 0.1
    friction: float = 0.05

    # Control
    position_command: float = 0.0
    velocity_command: float = 0.0
    torque_command: float = 0.0

    def get_transform(self) -> NDArray[np.float64]:
        """Get joint transformation matrix."""
        theta = self.position if self.joint_type == JointType.REVOLUTE else 0.0
        d = self.d + (self.position if self.joint_type == JointType.PRISMATIC else 0.0)

        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)

        T = np.array([
            [ct, -st * ca, st * sa, self.a * ct],
            [st, ct * ca, -ct * sa, self.a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

        return T


@dataclass
class SimulatedLink:
    """Simulated robot link."""
    name: str
    mass: float = 1.0

    # Geometry
    length: float = 0.1
    radius: float = 0.02

    # Inertia (diagonal)
    inertia: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01])
    )

    # Center of mass (relative to link frame)
    com: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )

    # Visual
    color: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.7, 0.7, 0.8, 1.0])
    )
    mesh_path: Optional[str] = None

    # Transform relative to parent
    transform: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(4)
    )


@dataclass
class SimulatedEndEffector:
    """Simulated end effector."""
    name: str
    ee_type: str = "gripper"  # gripper, tool, camera

    # Pose relative to last link
    local_pose: Pose = field(default_factory=Pose.identity)

    # Tool properties
    jaw_opening: float = 0.0  # For grippers
    jaw_max: float = 0.01  # 10mm
    grasp_force: float = 0.0

    # Tool tip
    tool_tip_offset: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0, 0, 0.05])
    )

    # Contact sensing
    contact_force: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(6)
    )
    in_contact: bool = False
    contact_body: Optional[str] = None

    # Actuation state
    actuated: bool = False
    actuation_value: float = 0.0

    def get_tool_tip_pose(self, link_pose: Pose) -> Pose:
        """Get world pose of tool tip."""
        # Combine link pose with local offset
        tip_position = link_pose.position + self._rotate_vector(
            link_pose.orientation,
            self.tool_tip_offset
        )
        return Pose(position=tip_position, orientation=link_pose.orientation)

    @staticmethod
    def _rotate_vector(q: NDArray, v: NDArray) -> NDArray:
        """Rotate vector by quaternion."""
        qw, qx, qy, qz = q

        # Quaternion rotation
        t = 2 * np.cross(np.array([qx, qy, qz]), v)
        return v + qw * t + np.cross(np.array([qx, qy, qz]), t)


class RobotSimulator:
    """
    Complete robot simulation for surgical robotics.

    Features:
    - Kinematic simulation
    - Dynamic simulation
    - Joint control (position, velocity, torque)
    - Collision detection
    - Force feedback
    """

    def __init__(
        self,
        name: str = "robot",
        num_joints: int = 7
    ) -> None:
        self.name = name
        self.num_joints = num_joints

        # Robot components
        self.joints: Dict[str, SimulatedJoint] = {}
        self.links: Dict[str, SimulatedLink] = {}
        self.end_effectors: Dict[str, SimulatedEndEffector] = {}

        # Base pose
        self.base_pose = Pose.identity()

        # State
        self._joint_positions = np.zeros(num_joints)
        self._joint_velocities = np.zeros(num_joints)
        self._joint_torques = np.zeros(num_joints)

        # Control mode
        self.control_mode = "position"  # position, velocity, torque

        # Simulation parameters
        self.dt = 0.001

        # Callbacks
        self._on_joint_limit: Optional[Callable[[str, float], None]] = None
        self._on_collision: Optional[Callable[[str, NDArray], None]] = None

        # Initialize default robot
        self._create_default_robot()

    def _create_default_robot(self) -> None:
        """Create default 7-DOF robot configuration."""
        # Typical surgical robot DH parameters
        dh_params = [
            {'d': 0.1, 'a': 0, 'alpha': -np.pi / 2},
            {'d': 0, 'a': 0, 'alpha': np.pi / 2},
            {'d': 0.2, 'a': 0, 'alpha': -np.pi / 2},
            {'d': 0, 'a': 0, 'alpha': np.pi / 2},
            {'d': 0.2, 'a': 0, 'alpha': -np.pi / 2},
            {'d': 0, 'a': 0, 'alpha': np.pi / 2},
            {'d': 0.1, 'a': 0, 'alpha': 0},
        ]

        for i in range(self.num_joints):
            params = dh_params[i] if i < len(dh_params) else {}

            joint = SimulatedJoint(
                name=f"joint_{i}",
                joint_type=JointType.REVOLUTE,
                d=params.get('d', 0.1),
                a=params.get('a', 0),
                alpha=params.get('alpha', 0),
                position_min=-2.5,
                position_max=2.5,
                velocity_max=2.0,
                torque_max=80.0,
                parent_link=f"link_{i - 1}" if i > 0 else "base",
                child_link=f"link_{i}"
            )
            self.joints[joint.name] = joint

            link = SimulatedLink(
                name=f"link_{i}",
                mass=2.0 - i * 0.2,  # Decreasing mass toward tip
                length=params.get('d', 0.1) + params.get('a', 0)
            )
            self.links[link.name] = link

        # Add end effector
        self.end_effectors["gripper"] = SimulatedEndEffector(
            name="gripper",
            ee_type="gripper",
            jaw_max=0.012,
            tool_tip_offset=np.array([0, 0, 0.06])
        )

    def set_joint_positions(self, positions: NDArray[np.float64]) -> None:
        """Set all joint positions."""
        positions = np.asarray(positions)
        for i, (name, joint) in enumerate(self.joints.items()):
            if i < len(positions):
                joint.position_command = positions[i]

    def set_joint_velocities(self, velocities: NDArray[np.float64]) -> None:
        """Set all joint velocities."""
        velocities = np.asarray(velocities)
        for i, (name, joint) in enumerate(self.joints.items()):
            if i < len(velocities):
                joint.velocity_command = velocities[i]

    def set_joint_torques(self, torques: NDArray[np.float64]) -> None:
        """Set all joint torques."""
        torques = np.asarray(torques)
        for i, (name, joint) in enumerate(self.joints.items()):
            if i < len(torques):
                joint.torque_command = torques[i]

    def get_joint_positions(self) -> NDArray[np.float64]:
        """Get all joint positions."""
        return np.array([j.position for j in self.joints.values()])

    def get_joint_velocities(self) -> NDArray[np.float64]:
        """Get all joint velocities."""
        return np.array([j.velocity for j in self.joints.values()])

    def get_joint_torques(self) -> NDArray[np.float64]:
        """Get measured joint torques."""
        return np.array([j.torque for j in self.joints.values()])

    def step(self, dt: Optional[float] = None) -> None:
        """Step simulation forward."""
        dt = dt or self.dt

        for name, joint in self.joints.items():
            if joint.joint_type == JointType.FIXED:
                continue

            if self.control_mode == "position":
                self._step_position_control(joint, dt)
            elif self.control_mode == "velocity":
                self._step_velocity_control(joint, dt)
            elif self.control_mode == "torque":
                self._step_torque_control(joint, dt)

            # Apply joint limits
            self._apply_joint_limits(joint)

    def _step_position_control(
        self,
        joint: SimulatedJoint,
        dt: float
    ) -> None:
        """Position control with PD gains."""
        kp = 100.0  # Position gain
        kd = 20.0  # Derivative gain

        # PD control
        position_error = joint.position_command - joint.position
        velocity_error = -joint.velocity

        torque = kp * position_error + kd * velocity_error

        # Apply torque limit
        torque = np.clip(torque, -joint.torque_max, joint.torque_max)

        # Dynamics
        acceleration = (torque - joint.damping * joint.velocity) / joint.inertia

        joint.velocity += acceleration * dt
        joint.position += joint.velocity * dt
        joint.torque = torque

    def _step_velocity_control(
        self,
        joint: SimulatedJoint,
        dt: float
    ) -> None:
        """Velocity control."""
        kv = 50.0  # Velocity gain

        velocity_error = joint.velocity_command - joint.velocity
        torque = kv * velocity_error

        torque = np.clip(torque, -joint.torque_max, joint.torque_max)

        acceleration = (torque - joint.damping * joint.velocity) / joint.inertia

        joint.velocity += acceleration * dt
        joint.position += joint.velocity * dt
        joint.torque = torque

    def _step_torque_control(
        self,
        joint: SimulatedJoint,
        dt: float
    ) -> None:
        """Direct torque control."""
        torque = np.clip(
            joint.torque_command,
            -joint.torque_max,
            joint.torque_max
        )

        acceleration = (torque - joint.damping * joint.velocity - joint.friction * np.sign(joint.velocity)) / joint.inertia

        joint.velocity += acceleration * dt
        joint.position += joint.velocity * dt
        joint.torque = torque

    def _apply_joint_limits(self, joint: SimulatedJoint) -> None:
        """Apply joint position and velocity limits."""
        # Position limits
        if joint.position < joint.position_min:
            joint.position = joint.position_min
            joint.velocity = max(0, joint.velocity)
            if self._on_joint_limit:
                self._on_joint_limit(joint.name, joint.position_min)

        elif joint.position > joint.position_max:
            joint.position = joint.position_max
            joint.velocity = min(0, joint.velocity)
            if self._on_joint_limit:
                self._on_joint_limit(joint.name, joint.position_max)

        # Velocity limits
        joint.velocity = np.clip(
            joint.velocity,
            -joint.velocity_max,
            joint.velocity_max
        )

    def forward_kinematics(
        self,
        joint_positions: Optional[NDArray[np.float64]] = None
    ) -> Pose:
        """Compute end-effector pose from joint positions."""
        if joint_positions is not None:
            for i, (name, joint) in enumerate(self.joints.items()):
                if i < len(joint_positions):
                    joint.position = joint_positions[i]

        # Compute chain of transforms
        T = self._pose_to_matrix(self.base_pose)

        for joint in self.joints.values():
            T = T @ joint.get_transform()

        # Extract pose
        position = T[:3, 3]
        orientation = self._matrix_to_quaternion(T[:3, :3])

        return Pose(position=position, orientation=orientation)

    def get_end_effector_pose(self) -> Pose:
        """Get current end-effector pose."""
        return self.forward_kinematics()

    def get_jacobian(
        self,
        joint_positions: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Compute Jacobian matrix at current/given configuration."""
        if joint_positions is None:
            joint_positions = self.get_joint_positions()

        n = len(joint_positions)
        J = np.zeros((6, n))

        # Numerical Jacobian
        eps = 1e-6
        ee_pose = self.forward_kinematics(joint_positions)

        for i in range(n):
            q_plus = joint_positions.copy()
            q_plus[i] += eps

            pose_plus = self.forward_kinematics(q_plus)

            # Position Jacobian
            J[:3, i] = (pose_plus.position - ee_pose.position) / eps

            # Orientation Jacobian (simplified)
            J[3:, i] = (pose_plus.orientation[1:] - ee_pose.orientation[1:]) / eps

        return J

    def inverse_kinematics(
        self,
        target_pose: Pose,
        initial_guess: Optional[NDArray[np.float64]] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ) -> Tuple[NDArray[np.float64], bool]:
        """
        Compute joint positions for target end-effector pose.

        Returns (joint_positions, success).
        """
        if initial_guess is None:
            q = self.get_joint_positions()
        else:
            q = initial_guess.copy()

        for iteration in range(max_iterations):
            current_pose = self.forward_kinematics(q)

            # Position error
            pos_error = target_pose.position - current_pose.position

            # Orientation error (simplified)
            orn_error = target_pose.orientation[1:] - current_pose.orientation[1:]

            error = np.concatenate([pos_error, orn_error])
            error_norm = np.linalg.norm(error)

            if error_norm < tolerance:
                return q, True

            # Jacobian
            J = self.get_jacobian(q)

            # Damped least squares
            lambda_dls = 0.01
            JtJ = J.T @ J
            dq = np.linalg.solve(
                JtJ + lambda_dls * np.eye(len(q)),
                J.T @ error
            )

            # Update
            q = q + 0.5 * dq

            # Apply joint limits
            for i, joint in enumerate(self.joints.values()):
                q[i] = np.clip(q[i], joint.position_min, joint.position_max)

        return q, False

    def apply_external_force(
        self,
        force: NDArray[np.float64],
        position: Optional[NDArray[np.float64]] = None
    ) -> None:
        """Apply external force to end-effector."""
        if position is None:
            position = self.get_end_effector_pose().position

        # Compute joint torques from external force using Jacobian transpose
        J = self.get_jacobian()
        joint_torques = J.T @ force

        for i, joint in enumerate(self.joints.values()):
            if i < len(joint_torques):
                joint.torque += joint_torques[i]

    def set_end_effector_actuation(
        self,
        ee_name: str,
        value: float
    ) -> None:
        """Set end-effector actuation (e.g., gripper opening)."""
        if ee_name in self.end_effectors:
            ee = self.end_effectors[ee_name]
            ee.actuation_value = np.clip(value, 0, 1)
            ee.jaw_opening = ee.actuation_value * ee.jaw_max
            ee.actuated = True

    def get_contact_force(self, ee_name: str = "gripper") -> NDArray[np.float64]:
        """Get contact force at end-effector."""
        if ee_name in self.end_effectors:
            return self.end_effectors[ee_name].contact_force.copy()
        return np.zeros(6)

    def set_contact_force(
        self,
        ee_name: str,
        force: NDArray[np.float64]
    ) -> None:
        """Set contact force (from collision detection)."""
        if ee_name in self.end_effectors:
            self.end_effectors[ee_name].contact_force = force.copy()
            self.end_effectors[ee_name].in_contact = np.linalg.norm(force) > 0.1

    def get_state(self) -> Dict[str, Any]:
        """Get complete robot state."""
        return {
            'joint_positions': self.get_joint_positions(),
            'joint_velocities': self.get_joint_velocities(),
            'joint_torques': self.get_joint_torques(),
            'ee_pose': self.get_end_effector_pose(),
            'control_mode': self.control_mode,
            'end_effectors': {
                name: {
                    'jaw_opening': ee.jaw_opening,
                    'contact_force': ee.contact_force.tolist(),
                    'in_contact': ee.in_contact
                }
                for name, ee in self.end_effectors.items()
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set robot state."""
        if 'joint_positions' in state:
            positions = state['joint_positions']
            for i, joint in enumerate(self.joints.values()):
                if i < len(positions):
                    joint.position = positions[i]

        if 'joint_velocities' in state:
            velocities = state['joint_velocities']
            for i, joint in enumerate(self.joints.values()):
                if i < len(velocities):
                    joint.velocity = velocities[i]

    def reset(self) -> None:
        """Reset robot to initial state."""
        for joint in self.joints.values():
            joint.position = 0.0
            joint.velocity = 0.0
            joint.torque = 0.0
            joint.position_command = 0.0
            joint.velocity_command = 0.0
            joint.torque_command = 0.0

        for ee in self.end_effectors.values():
            ee.jaw_opening = 0.0
            ee.contact_force = np.zeros(6)
            ee.in_contact = False

    @staticmethod
    def _pose_to_matrix(pose: Pose) -> NDArray[np.float64]:
        """Convert pose to 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, 3] = pose.position

        # Quaternion to rotation matrix
        w, x, y, z = pose.orientation
        T[0, 0] = 1 - 2 * (y * y + z * z)
        T[0, 1] = 2 * (x * y - z * w)
        T[0, 2] = 2 * (x * z + y * w)
        T[1, 0] = 2 * (x * y + z * w)
        T[1, 1] = 1 - 2 * (x * x + z * z)
        T[1, 2] = 2 * (y * z - x * w)
        T[2, 0] = 2 * (x * z - y * w)
        T[2, 1] = 2 * (y * z + x * w)
        T[2, 2] = 1 - 2 * (x * x + y * y)

        return T

    @staticmethod
    def _matrix_to_quaternion(R: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert rotation matrix to quaternion."""
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
