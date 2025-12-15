"""Impedance and Admittance Control for surgical robotics."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import Pose


class ControlMode(Enum):
    """Interaction control modes."""
    POSITION = auto()
    FORCE = auto()
    IMPEDANCE = auto()
    ADMITTANCE = auto()
    HYBRID = auto()


@dataclass
class ImpedanceParams:
    """Impedance control parameters (virtual spring-mass-damper)."""
    # Stiffness (N/m for translation, Nm/rad for rotation)
    stiffness: NDArray[np.float64]  # 6D: [Kx, Ky, Kz, Krx, Kry, Krz]

    # Damping (Ns/m for translation, Nms/rad for rotation)
    damping: NDArray[np.float64]  # 6D

    # Inertia (kg for translation, kg*m^2 for rotation)
    inertia: NDArray[np.float64]  # 6D

    @classmethod
    def default(cls) -> "ImpedanceParams":
        """Create default impedance parameters."""
        return cls(
            stiffness=np.array([500, 500, 500, 20, 20, 20]),
            damping=np.array([50, 50, 50, 2, 2, 2]),
            inertia=np.array([1, 1, 1, 0.1, 0.1, 0.1])
        )

    @classmethod
    def soft(cls) -> "ImpedanceParams":
        """Create soft impedance for delicate operations."""
        return cls(
            stiffness=np.array([100, 100, 100, 5, 5, 5]),
            damping=np.array([20, 20, 20, 1, 1, 1]),
            inertia=np.array([0.5, 0.5, 0.5, 0.05, 0.05, 0.05])
        )

    @classmethod
    def stiff(cls) -> "ImpedanceParams":
        """Create stiff impedance for precise positioning."""
        return cls(
            stiffness=np.array([2000, 2000, 2000, 100, 100, 100]),
            damping=np.array([100, 100, 100, 10, 10, 10]),
            inertia=np.array([2, 2, 2, 0.2, 0.2, 0.2])
        )


class ImpedanceController:
    """
    Impedance controller for compliant robot behavior.

    Implements: M*xdd + D*xd + K*(x - x_d) = F_ext

    Where:
    - M: Virtual inertia
    - D: Virtual damping
    - K: Virtual stiffness
    - x: Actual position
    - x_d: Desired position
    - F_ext: External force
    """

    def __init__(self, params: Optional[ImpedanceParams] = None) -> None:
        self.params = params or ImpedanceParams.default()

        # State
        self.desired_pose: Optional[Pose] = None
        self.desired_velocity = np.zeros(6)
        self.desired_acceleration = np.zeros(6)

        # Computed values
        self._last_velocity = np.zeros(6)
        self._last_position = np.zeros(6)

    def set_desired_pose(self, pose: Pose) -> None:
        """Set desired equilibrium pose."""
        self.desired_pose = pose

    def set_desired_trajectory(
        self,
        pose: Pose,
        velocity: NDArray[np.float64],
        acceleration: NDArray[np.float64]
    ) -> None:
        """Set desired trajectory point."""
        self.desired_pose = pose
        self.desired_velocity = velocity
        self.desired_acceleration = acceleration

    def compute_force(
        self,
        current_pose: Pose,
        current_velocity: NDArray[np.float64],
        external_force: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Compute required force/torque for impedance behavior.

        Args:
            current_pose: Current end-effector pose
            current_velocity: Current Cartesian velocity [vx,vy,vz,wx,wy,wz]
            external_force: Measured external force (optional)

        Returns:
            Required force/torque [Fx,Fy,Fz,Tx,Ty,Tz]
        """
        if self.desired_pose is None:
            return np.zeros(6)

        if external_force is None:
            external_force = np.zeros(6)

        # Position error
        position_error = current_pose.position - self.desired_pose.position

        # Orientation error (simplified - using quaternion difference)
        orientation_error = self._quaternion_error(
            current_pose.orientation,
            self.desired_pose.orientation
        )

        # Full 6D error
        pose_error = np.concatenate([position_error, orientation_error])

        # Velocity error
        velocity_error = current_velocity - self.desired_velocity

        # Impedance equation: F = M*xdd_d + D*(xd_d - xd) + K*(x_d - x) + F_ext
        # Rearranged for force output

        K = self.params.stiffness
        D = self.params.damping
        M = self.params.inertia

        # Desired acceleration from trajectory
        xdd_d = self.desired_acceleration

        # Compute required force
        force = (
            M * xdd_d -
            D * velocity_error -
            K * pose_error +
            external_force
        )

        return force

    @staticmethod
    def _quaternion_error(
        q_current: NDArray[np.float64],
        q_desired: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute orientation error as rotation vector."""
        # q_error = q_desired * q_current^-1
        q_inv = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])

        w1, x1, y1, z1 = q_desired
        w2, x2, y2, z2 = q_inv

        q_error = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

        # Convert to rotation vector
        if q_error[0] < 0:
            q_error = -q_error

        return 2.0 * q_error[1:4]


class AdmittanceController:
    """
    Admittance controller for force-guided motion.

    Implements: x = F_ext / (M*s^2 + D*s + K)

    Converts measured forces to desired motion.
    """

    def __init__(self, params: Optional[ImpedanceParams] = None) -> None:
        self.params = params or ImpedanceParams.default()

        # State
        self.equilibrium_pose: Optional[Pose] = None

        # Integration state
        self._velocity = np.zeros(6)
        self._position_offset = np.zeros(6)

        # Filter state
        self._filtered_force = np.zeros(6)
        self._force_filter_alpha = 0.1

    def set_equilibrium_pose(self, pose: Pose) -> None:
        """Set equilibrium pose (zero-force position)."""
        self.equilibrium_pose = pose
        self._velocity = np.zeros(6)
        self._position_offset = np.zeros(6)

    def compute_motion(
        self,
        measured_force: NDArray[np.float64],
        dt: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute desired motion from measured force.

        Args:
            measured_force: Measured force/torque [Fx,Fy,Fz,Tx,Ty,Tz]
            dt: Time step

        Returns:
            Tuple of (position_offset, velocity)
        """
        # Low-pass filter force
        self._filtered_force = (
            self._force_filter_alpha * measured_force +
            (1 - self._force_filter_alpha) * self._filtered_force
        )

        K = self.params.stiffness
        D = self.params.damping
        M = self.params.inertia

        # Admittance dynamics: M*xdd + D*xd + K*x = F
        # Discretized with forward Euler

        # Compute acceleration
        acceleration = (
            self._filtered_force -
            D * self._velocity -
            K * self._position_offset
        ) / M

        # Integrate
        self._velocity += acceleration * dt
        self._position_offset += self._velocity * dt

        return self._position_offset.copy(), self._velocity.copy()

    def get_target_pose(self) -> Optional[Pose]:
        """Get target pose including offset from equilibrium."""
        if self.equilibrium_pose is None:
            return None

        target_position = self.equilibrium_pose.position + self._position_offset[:3]

        # Apply orientation offset (simplified)
        target_orientation = self.equilibrium_pose.orientation.copy()

        return Pose(position=target_position, orientation=target_orientation)

    def reset(self) -> None:
        """Reset controller state."""
        self._velocity = np.zeros(6)
        self._position_offset = np.zeros(6)
        self._filtered_force = np.zeros(6)


@dataclass
class HybridParams:
    """Parameters for hybrid force-position control."""
    # Selection matrix (1 = force control, 0 = position control)
    selection: NDArray[np.float64]  # 6D diagonal

    # Force control gains
    force_p_gain: float = 0.001
    force_i_gain: float = 0.0001
    force_d_gain: float = 0.0

    # Position control gains
    position_p_gain: float = 100.0
    position_d_gain: float = 20.0


class HybridForcePosition:
    """
    Hybrid force-position controller.

    Allows simultaneous force control in some directions
    and position control in others.
    """

    def __init__(self, params: Optional[HybridParams] = None) -> None:
        if params is None:
            # Default: force control in Z, position in X,Y
            params = HybridParams(
                selection=np.diag([0, 0, 1, 0, 0, 0])
            )
        self.params = params

        # Desired values
        self.desired_pose: Optional[Pose] = None
        self.desired_force = np.zeros(6)

        # Control state
        self._force_integral = np.zeros(6)
        self._last_force_error = np.zeros(6)

    def set_desired_pose(self, pose: Pose) -> None:
        """Set desired pose for position-controlled directions."""
        self.desired_pose = pose

    def set_desired_force(self, force: NDArray[np.float64]) -> None:
        """Set desired force for force-controlled directions."""
        self.desired_force = force

    def compute_control(
        self,
        current_pose: Pose,
        current_velocity: NDArray[np.float64],
        measured_force: NDArray[np.float64],
        dt: float
    ) -> NDArray[np.float64]:
        """
        Compute hybrid control output.

        Args:
            current_pose: Current end-effector pose
            current_velocity: Current velocity
            measured_force: Measured force/torque
            dt: Time step

        Returns:
            Control output (force/torque command)
        """
        if self.desired_pose is None:
            return np.zeros(6)

        S = self.params.selection  # Selection matrix

        # Position control output
        position_error = self._compute_pose_error(current_pose)
        position_output = (
            self.params.position_p_gain * position_error -
            self.params.position_d_gain * current_velocity
        )

        # Force control output (PID)
        force_error = self.desired_force - measured_force
        self._force_integral += force_error * dt

        force_output = (
            self.params.force_p_gain * force_error +
            self.params.force_i_gain * self._force_integral +
            self.params.force_d_gain * (force_error - self._last_force_error) / dt
        )
        self._last_force_error = force_error

        # Combine using selection matrix
        # Force-controlled directions use force output
        # Position-controlled directions use position output
        I = np.eye(6)
        output = S @ force_output + (I - S) @ position_output

        return output

    def _compute_pose_error(self, current_pose: Pose) -> NDArray[np.float64]:
        """Compute 6D pose error."""
        position_error = self.desired_pose.position - current_pose.position

        # Orientation error (simplified)
        orientation_error = np.zeros(3)

        return np.concatenate([position_error, orientation_error])

    def set_selection_matrix(self, selection: NDArray[np.float64]) -> None:
        """
        Set selection matrix for hybrid control.

        Args:
            selection: 6x6 diagonal matrix (1=force, 0=position)
        """
        self.params.selection = selection

    def reset(self) -> None:
        """Reset controller state."""
        self._force_integral = np.zeros(6)
        self._last_force_error = np.zeros(6)


class VariableImpedanceController:
    """
    Variable impedance controller that adapts parameters
    based on task requirements or sensed information.
    """

    def __init__(self) -> None:
        self.base_params = ImpedanceParams.default()
        self.current_params = ImpedanceParams.default()
        self.impedance_controller = ImpedanceController(self.current_params)

        # Adaptation parameters
        self.adaptation_rate = 0.1
        self.min_stiffness = np.array([50, 50, 50, 1, 1, 1])
        self.max_stiffness = np.array([5000, 5000, 5000, 200, 200, 200])

    def adapt_to_force(self, measured_force: NDArray[np.float64]) -> None:
        """
        Adapt impedance based on measured forces.

        Reduces stiffness when high forces are detected.
        """
        force_magnitude = np.abs(measured_force)

        # Reduce stiffness in directions with high force
        threshold = 10.0  # Newtons
        scale = np.exp(-force_magnitude / threshold)

        new_stiffness = self.base_params.stiffness * scale
        new_stiffness = np.clip(new_stiffness, self.min_stiffness, self.max_stiffness)

        # Smooth adaptation
        self.current_params.stiffness = (
            (1 - self.adaptation_rate) * self.current_params.stiffness +
            self.adaptation_rate * new_stiffness
        )

        # Update damping to maintain critical damping ratio
        self.current_params.damping = 2 * np.sqrt(
            self.current_params.stiffness * self.current_params.inertia
        )

        self.impedance_controller.params = self.current_params

    def adapt_to_task(self, task_phase: str) -> None:
        """
        Adapt impedance based on task phase.

        Args:
            task_phase: Task identifier (e.g., "approach", "contact", "insertion")
        """
        if task_phase == "approach":
            self.current_params = ImpedanceParams.stiff()
        elif task_phase == "contact":
            self.current_params = ImpedanceParams.default()
        elif task_phase == "insertion":
            self.current_params = ImpedanceParams.soft()

        self.impedance_controller.params = self.current_params

    def compute_force(
        self,
        current_pose: Pose,
        current_velocity: NDArray[np.float64],
        external_force: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Compute force using current impedance parameters."""
        return self.impedance_controller.compute_force(
            current_pose, current_velocity, external_force
        )
