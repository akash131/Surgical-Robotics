"""Kinematics computations for surgical robots."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import Pose, RobotArm


@dataclass
class DHParameters:
    """Denavit-Hartenberg parameters for a single link."""
    theta: float  # Joint angle (rad)
    d: float  # Link offset (m)
    a: float  # Link length (m)
    alpha: float  # Link twist (rad)

    def transformation_matrix(self, joint_offset: float = 0.0) -> NDArray[np.float64]:
        """Compute transformation matrix for this link."""
        theta = self.theta + joint_offset
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)

        return np.array([
            [ct, -st * ca, st * sa, self.a * ct],
            [st, ct * ca, -ct * sa, self.a * st],
            [0, sa, ca, self.d],
            [0, 0, 0, 1]
        ])


class ForwardKinematics:
    """Forward kinematics solver using DH convention."""

    def __init__(self, dh_params: list[DHParameters]) -> None:
        self.dh_params = dh_params
        self.num_joints = len(dh_params)

    def compute(self, joint_positions: NDArray[np.float64]) -> Pose:
        """
        Compute end-effector pose from joint positions.

        Args:
            joint_positions: Array of joint angles/positions

        Returns:
            End-effector pose
        """
        if len(joint_positions) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {len(joint_positions)}")

        transform = np.eye(4)
        for i, (dh, q) in enumerate(zip(self.dh_params, joint_positions)):
            transform = transform @ dh.transformation_matrix(q)

        position = transform[:3, 3]
        rotation_matrix = transform[:3, :3]
        orientation = self._rotation_matrix_to_quaternion(rotation_matrix)

        return Pose(position=position, orientation=orientation)

    def compute_all_frames(
        self, joint_positions: NDArray[np.float64]
    ) -> list[NDArray[np.float64]]:
        """Compute transformation matrices for all joint frames."""
        frames = [np.eye(4)]
        transform = np.eye(4)

        for dh, q in zip(self.dh_params, joint_positions):
            transform = transform @ dh.transformation_matrix(q)
            frames.append(transform.copy())

        return frames

    @staticmethod
    def _rotation_matrix_to_quaternion(R: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert rotation matrix to quaternion [w, x, y, z]."""
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


class InverseKinematics:
    """Inverse kinematics solver using numerical methods."""

    def __init__(
        self,
        forward_kinematics: ForwardKinematics,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        damping: float = 0.01
    ) -> None:
        self.fk = forward_kinematics
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping

    def compute(
        self,
        target_pose: Pose,
        initial_guess: Optional[NDArray[np.float64]] = None,
        joint_limits: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None
    ) -> Tuple[NDArray[np.float64], bool]:
        """
        Compute joint positions for target end-effector pose.

        Uses damped least squares (Levenberg-Marquardt) method.

        Args:
            target_pose: Desired end-effector pose
            initial_guess: Starting joint configuration
            joint_limits: Tuple of (min_limits, max_limits) arrays

        Returns:
            Tuple of (joint_positions, success_flag)
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.fk.num_joints)

        q = initial_guess.copy()
        target_position = target_pose.position
        target_orientation = target_pose.orientation

        for iteration in range(self.max_iterations):
            current_pose = self.fk.compute(q)

            # Position error
            pos_error = target_position - current_pose.position

            # Orientation error (quaternion)
            orient_error = self._quaternion_error(
                target_orientation, current_pose.orientation
            )

            # Combined error
            error = np.concatenate([pos_error, orient_error])
            error_norm = np.linalg.norm(error)

            if error_norm < self.tolerance:
                return q, True

            # Compute Jacobian
            J = self._compute_jacobian(q)

            # Damped least squares solution
            JtJ = J.T @ J
            damping_matrix = self.damping * np.eye(self.fk.num_joints)
            delta_q = np.linalg.solve(JtJ + damping_matrix, J.T @ error)

            # Update joints
            q = q + delta_q

            # Apply joint limits if provided
            if joint_limits is not None:
                q = np.clip(q, joint_limits[0], joint_limits[1])

        return q, False

    def _compute_jacobian(
        self, q: NDArray[np.float64], delta: float = 1e-6
    ) -> NDArray[np.float64]:
        """Compute Jacobian matrix using numerical differentiation."""
        J = np.zeros((6, self.fk.num_joints))

        base_pose = self.fk.compute(q)
        base_pos = base_pose.position
        base_orient = base_pose.orientation

        for i in range(self.fk.num_joints):
            q_plus = q.copy()
            q_plus[i] += delta

            pose_plus = self.fk.compute(q_plus)

            # Position derivative
            J[:3, i] = (pose_plus.position - base_pos) / delta

            # Orientation derivative
            J[3:6, i] = self._quaternion_derivative(
                base_orient, pose_plus.orientation, delta
            )

        return J

    @staticmethod
    def _quaternion_error(
        target: NDArray[np.float64], current: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute orientation error as rotation vector."""
        # Quaternion multiplication: target * inverse(current)
        current_inv = np.array([current[0], -current[1], -current[2], -current[3]])

        w1, x1, y1, z1 = target
        w2, x2, y2, z2 = current_inv

        error_quat = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

        # Convert to rotation vector (axis-angle)
        if error_quat[0] < 0:
            error_quat = -error_quat

        return 2.0 * error_quat[1:4]

    @staticmethod
    def _quaternion_derivative(
        q1: NDArray[np.float64], q2: NDArray[np.float64], delta: float
    ) -> NDArray[np.float64]:
        """Compute derivative of quaternion."""
        return (q2[1:4] - q1[1:4]) / delta


class TrajectoryPlanner:
    """Plan smooth trajectories in joint or Cartesian space."""

    @staticmethod
    def linear_interpolation(
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        num_points: int
    ) -> NDArray[np.float64]:
        """Linear interpolation between two configurations."""
        t = np.linspace(0, 1, num_points)
        return np.outer(1 - t, start) + np.outer(t, end)

    @staticmethod
    def quintic_polynomial(
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        duration: float,
        dt: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate quintic polynomial trajectory (zero velocity/acceleration at endpoints).

        Returns:
            Tuple of (positions, velocities, accelerations)
        """
        num_points = int(duration / dt) + 1
        t = np.linspace(0, duration, num_points)
        T = duration

        # Quintic polynomial coefficients for s(t) from 0 to 1
        # s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # With boundary conditions: s(0)=0, s(T)=1, s'(0)=s'(T)=0, s''(0)=s''(T)=0
        a0 = 0
        a1 = 0
        a2 = 0
        a3 = 10 / T**3
        a4 = -15 / T**4
        a5 = 6 / T**5

        s = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        s_dot = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        s_ddot = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3

        diff = end - start
        positions = np.outer(s, diff) + start
        velocities = np.outer(s_dot, diff)
        accelerations = np.outer(s_ddot, diff)

        return positions, velocities, accelerations
