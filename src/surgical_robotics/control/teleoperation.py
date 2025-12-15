"""Teleoperation control for surgical robotics."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, List
import numpy as np
from numpy.typing import NDArray

from surgical_robotics.core.base import Pose


class TeleoperationMode(Enum):
    """Teleoperation modes."""
    POSITION = auto()  # Direct position control
    VELOCITY = auto()  # Velocity control
    INCREMENTAL = auto()  # Incremental position
    BILATERAL = auto()  # Bilateral with force feedback


@dataclass
class TeleoperationConfig:
    """Configuration for teleoperation."""
    mode: TeleoperationMode = TeleoperationMode.POSITION

    # Motion scaling
    position_scale: float = 0.5  # Master to slave position ratio
    orientation_scale: float = 0.5
    velocity_scale: float = 0.5

    # Workspace mapping
    master_workspace_radius: float = 0.15  # meters
    slave_workspace_radius: float = 0.3

    # Filtering
    position_filter_alpha: float = 0.3  # Low-pass filter coefficient
    velocity_filter_alpha: float = 0.5

    # Deadband
    position_deadband: float = 0.001  # 1mm
    orientation_deadband: float = 0.01  # ~0.5 degrees

    # Rate limits
    max_linear_velocity: float = 0.2  # m/s
    max_angular_velocity: float = 1.0  # rad/s


class MotionScaler:
    """
    Scales motion between master and slave robots.

    Handles workspace mapping, scaling, and filtering.
    """

    def __init__(self, config: Optional[TeleoperationConfig] = None) -> None:
        self.config = config or TeleoperationConfig()

        # Reference poses
        self._master_reference: Optional[Pose] = None
        self._slave_reference: Optional[Pose] = None

        # Filtered values
        self._filtered_position = np.zeros(3)
        self._filtered_orientation = np.array([1, 0, 0, 0])

    def set_reference(self, master_pose: Pose, slave_pose: Pose) -> None:
        """Set reference (clutch) position."""
        self._master_reference = Pose(
            position=master_pose.position.copy(),
            orientation=master_pose.orientation.copy()
        )
        self._slave_reference = Pose(
            position=slave_pose.position.copy(),
            orientation=slave_pose.orientation.copy()
        )
        self._filtered_position = slave_pose.position.copy()
        self._filtered_orientation = slave_pose.orientation.copy()

    def scale_motion(
        self,
        master_pose: Pose,
        dt: float
    ) -> Pose:
        """
        Scale master motion to slave.

        Args:
            master_pose: Current master pose
            dt: Time step

        Returns:
            Target slave pose
        """
        if self._master_reference is None or self._slave_reference is None:
            return Pose.identity()

        # Compute master displacement from reference
        master_delta = master_pose.position - self._master_reference.position

        # Apply scaling
        scaled_delta = master_delta * self.config.position_scale

        # Apply deadband
        delta_magnitude = float(np.linalg.norm(scaled_delta))
        if delta_magnitude < self.config.position_deadband:
            scaled_delta = np.zeros(3)
        else:
            # Remove deadband region
            scaled_delta = scaled_delta * (
                (delta_magnitude - self.config.position_deadband) / delta_magnitude
            )

        # Compute target position
        target_position = self._slave_reference.position + scaled_delta

        # Apply velocity limit
        position_change = target_position - self._filtered_position
        velocity = position_change / dt if dt > 0 else np.zeros(3)
        speed = float(np.linalg.norm(velocity))

        if speed > self.config.max_linear_velocity:
            velocity = velocity / speed * self.config.max_linear_velocity
            target_position = self._filtered_position + velocity * dt

        # Low-pass filter
        alpha = self.config.position_filter_alpha
        self._filtered_position = (
            alpha * target_position +
            (1 - alpha) * self._filtered_position
        )

        # Scale orientation (simplified - would use proper quaternion interpolation)
        target_orientation = master_pose.orientation.copy()
        self._filtered_orientation = self._slerp(
            self._filtered_orientation,
            target_orientation,
            self.config.orientation_scale
        )

        return Pose(
            position=self._filtered_position.copy(),
            orientation=self._filtered_orientation.copy()
        )

    @staticmethod
    def _slerp(
        q1: NDArray[np.float64],
        q2: NDArray[np.float64],
        t: float
    ) -> NDArray[np.float64]:
        """Spherical linear interpolation between quaternions."""
        dot = np.dot(q1, q2)

        if dot < 0:
            q2 = -q2
            dot = -dot

        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        return s1 * q1 + s2 * q2


class TimeDelayCompensator:
    """
    Compensates for communication time delays in teleoperation.

    Implements wave variable transformation for stability.
    """

    def __init__(
        self,
        impedance: float = 50.0,  # Wave impedance
        buffer_size: int = 100
    ) -> None:
        self.impedance = impedance
        self.buffer_size = buffer_size

        # Wave variable buffers
        self._outgoing_waves: List[tuple] = []  # (timestamp, wave_value)
        self._incoming_waves: List[tuple] = []

        # State
        self._last_velocity = np.zeros(6)
        self._last_force = np.zeros(6)

    def encode_to_wave(
        self,
        velocity: NDArray[np.float64],
        force: NDArray[np.float64],
        timestamp: float
    ) -> NDArray[np.float64]:
        """
        Encode velocity and force to wave variable.

        Wave variable: u = (b*v + f) / sqrt(2*b)
        where b is wave impedance, v is velocity, f is force
        """
        b = self.impedance
        wave = (b * velocity + force) / np.sqrt(2 * b)

        self._outgoing_waves.append((timestamp, wave.copy()))

        # Trim buffer
        if len(self._outgoing_waves) > self.buffer_size:
            self._outgoing_waves.pop(0)

        return wave

    def decode_from_wave(
        self,
        wave: NDArray[np.float64],
        timestamp: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Decode wave variable to velocity and force.

        Returns:
            Tuple of (velocity, force)
        """
        self._incoming_waves.append((timestamp, wave.copy()))

        if len(self._incoming_waves) > self.buffer_size:
            self._incoming_waves.pop(0)

        b = self.impedance

        # Reconstruct velocity and force
        # v = (u_in + u_out) / (sqrt(2*b))
        # f = sqrt(2*b) * (u_in - u_out) / 2

        # Get corresponding outgoing wave
        if self._outgoing_waves:
            _, u_out = self._outgoing_waves[-1]
        else:
            u_out = np.zeros_like(wave)

        velocity = (wave + u_out) / np.sqrt(2 * b)
        force = np.sqrt(2 * b) * (wave - u_out) / 2

        self._last_velocity = velocity
        self._last_force = force

        return velocity, force

    def get_delay_estimate(self) -> float:
        """Estimate current communication delay."""
        if len(self._incoming_waves) < 2:
            return 0.0

        # Simple delay estimation from timestamps
        recent_timestamps = [t for t, _ in self._incoming_waves[-10:]]
        if len(recent_timestamps) < 2:
            return 0.0

        intervals = np.diff(recent_timestamps)
        return float(np.mean(intervals))


class BilateralController:
    """
    Bilateral teleoperation controller with force feedback.

    Implements 4-channel architecture for stable bilateral control.
    """

    def __init__(
        self,
        config: Optional[TeleoperationConfig] = None
    ) -> None:
        self.config = config or TeleoperationConfig()
        self.config.mode = TeleoperationMode.BILATERAL

        self.motion_scaler = MotionScaler(self.config)
        self.delay_compensator = TimeDelayCompensator()

        # Controller gains
        self.position_gain = 500.0  # N/m
        self.velocity_gain = 50.0  # Ns/m
        self.force_scale = 1.0

        # State
        self._master_pose: Optional[Pose] = None
        self._slave_pose: Optional[Pose] = None
        self._slave_force = np.zeros(6)

    def update_master(
        self,
        master_pose: Pose,
        master_velocity: NDArray[np.float64]
    ) -> tuple[Pose, NDArray[np.float64]]:
        """
        Update from master side.

        Returns:
            Tuple of (slave_command_pose, master_force_feedback)
        """
        self._master_pose = master_pose

        # Compute slave command
        slave_target = self.motion_scaler.scale_motion(master_pose, 0.001)

        # Compute force feedback for master
        if self._slave_pose is not None:
            # Position error force
            position_error = slave_target.position - self._slave_pose.position
            position_force = self.position_gain * position_error

            # Velocity damping (simplified)
            velocity_force = -self.velocity_gain * master_velocity[:3]

            # Add scaled slave contact force
            contact_force = self.force_scale * self._slave_force[:3]

            master_feedback = position_force + velocity_force + contact_force
        else:
            master_feedback = np.zeros(3)

        return slave_target, np.concatenate([master_feedback, np.zeros(3)])

    def update_slave(
        self,
        slave_pose: Pose,
        slave_force: NDArray[np.float64]
    ) -> None:
        """
        Update from slave side.

        Args:
            slave_pose: Current slave pose
            slave_force: Measured slave force/torque
        """
        self._slave_pose = slave_pose
        self._slave_force = slave_force

    def set_reference(self, master_pose: Pose, slave_pose: Pose) -> None:
        """Set clutch reference."""
        self.motion_scaler.set_reference(master_pose, slave_pose)


class TeleoperationController:
    """
    Complete teleoperation controller.

    Integrates motion scaling, time delay compensation,
    and bilateral control.
    """

    def __init__(
        self,
        config: Optional[TeleoperationConfig] = None
    ) -> None:
        self.config = config or TeleoperationConfig()

        # Sub-controllers
        self.motion_scaler = MotionScaler(self.config)
        self.delay_compensator = TimeDelayCompensator()
        self.bilateral_controller = BilateralController(self.config)

        # State
        self.clutch_engaged = False
        self.enabled = False

        # Workspace limits (for slave)
        self.workspace_center = np.array([0, 0, 0.3])
        self.workspace_radius = 0.25

        # Callbacks
        self._on_workspace_violation: Optional[Callable[[], None]] = None

    def enable(self) -> None:
        """Enable teleoperation."""
        self.enabled = True

    def disable(self) -> None:
        """Disable teleoperation."""
        self.enabled = False

    def engage_clutch(
        self,
        master_pose: Pose,
        slave_pose: Pose
    ) -> None:
        """Engage clutch (set reference poses)."""
        self.clutch_engaged = True
        self.motion_scaler.set_reference(master_pose, slave_pose)
        self.bilateral_controller.set_reference(master_pose, slave_pose)

    def release_clutch(self) -> None:
        """Release clutch."""
        self.clutch_engaged = False

    def update(
        self,
        master_pose: Pose,
        master_velocity: NDArray[np.float64],
        slave_pose: Pose,
        slave_force: NDArray[np.float64],
        dt: float
    ) -> tuple[Optional[Pose], NDArray[np.float64]]:
        """
        Main teleoperation update.

        Args:
            master_pose: Current master pose
            master_velocity: Current master velocity
            slave_pose: Current slave pose
            slave_force: Measured slave force
            dt: Time step

        Returns:
            Tuple of (slave_command, master_force_feedback)
        """
        if not self.enabled:
            return None, np.zeros(6)

        if self.clutch_engaged:
            return None, np.zeros(6)

        # Get slave command based on mode
        if self.config.mode == TeleoperationMode.POSITION:
            slave_command = self.motion_scaler.scale_motion(master_pose, dt)
            master_feedback = np.zeros(6)

        elif self.config.mode == TeleoperationMode.BILATERAL:
            self.bilateral_controller.update_slave(slave_pose, slave_force)
            slave_command, master_feedback = self.bilateral_controller.update_master(
                master_pose, master_velocity
            )

        else:
            slave_command = self.motion_scaler.scale_motion(master_pose, dt)
            master_feedback = np.zeros(6)

        # Check workspace limits
        if slave_command:
            slave_command = self._enforce_workspace_limits(slave_command)

        return slave_command, master_feedback

    def _enforce_workspace_limits(self, pose: Pose) -> Pose:
        """Enforce workspace limits on slave command."""
        position = pose.position.copy()

        # Check distance from workspace center
        offset = position - self.workspace_center
        distance = float(np.linalg.norm(offset))

        if distance > self.workspace_radius:
            # Project back to workspace boundary
            position = self.workspace_center + offset / distance * self.workspace_radius

            if self._on_workspace_violation:
                self._on_workspace_violation()

        return Pose(position=position, orientation=pose.orientation)

    def set_workspace_violation_callback(
        self,
        callback: Callable[[], None]
    ) -> None:
        """Set callback for workspace violations."""
        self._on_workspace_violation = callback

    def get_delay(self) -> float:
        """Get estimated communication delay."""
        return self.delay_compensator.get_delay_estimate()
