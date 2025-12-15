"""Haptic feedback systems for Da Vinci surgical robots."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray


class FeedbackType(Enum):
    """Type of haptic feedback."""
    FORCE = auto()
    VIBRATION = auto()
    TEXTURE = auto()
    TEMPERATURE = auto()


@dataclass
class ForceReading:
    """Force/torque sensor reading."""
    forces: NDArray[np.float64]  # [Fx, Fy, Fz] in Newtons
    torques: NDArray[np.float64]  # [Tx, Ty, Tz] in Nm
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.forces = np.asarray(self.forces, dtype=np.float64)
        self.torques = np.asarray(self.torques, dtype=np.float64)

    @property
    def force_magnitude(self) -> float:
        """Total force magnitude."""
        return float(np.linalg.norm(self.forces))

    @property
    def torque_magnitude(self) -> float:
        """Total torque magnitude."""
        return float(np.linalg.norm(self.torques))


@dataclass
class HapticState:
    """Current state of haptic device."""
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    force_output: NDArray[np.float64]
    is_grasped: bool = False
    button_states: dict[str, bool] = field(default_factory=dict)


class HapticDevice:
    """
    Haptic device interface for master console controllers.

    Provides force feedback to surgeon's hands based on
    instrument-tissue interaction forces.
    """

    def __init__(
        self,
        name: str,
        max_force: float = 5.0,
        workspace_radius: float = 0.15
    ) -> None:
        self.name = name
        self.max_force = max_force  # Maximum feedback force in Newtons
        self.workspace_radius = workspace_radius  # meters

        self.state = HapticState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            force_output=np.zeros(3)
        )

        self.force_scaling = 1.0
        self.deadband = 0.1  # Force deadband in Newtons
        self.is_enabled = False

        # Low-pass filter for force smoothing
        self._force_filter_alpha = 0.1
        self._filtered_force = np.zeros(3)

    def enable(self) -> None:
        """Enable haptic feedback."""
        self.is_enabled = True

    def disable(self) -> None:
        """Disable haptic feedback and zero forces."""
        self.is_enabled = False
        self.state.force_output = np.zeros(3)
        self._filtered_force = np.zeros(3)

    def update_position(self, position: NDArray[np.float64]) -> None:
        """Update device position reading."""
        prev_position = self.state.position.copy()
        self.state.position = np.asarray(position)

        # Estimate velocity (would normally use actual velocity sensor)
        dt = 0.001  # Assume 1kHz update rate
        self.state.velocity = (self.state.position - prev_position) / dt

    def set_force(self, force: NDArray[np.float64]) -> None:
        """
        Set output force to render on haptic device.

        Args:
            force: 3D force vector to render
        """
        if not self.is_enabled:
            self.state.force_output = np.zeros(3)
            return

        force = np.asarray(force, dtype=np.float64)

        # Apply scaling
        scaled_force = force * self.force_scaling

        # Apply deadband
        force_magnitude = float(np.linalg.norm(scaled_force))
        if force_magnitude < self.deadband:
            scaled_force = np.zeros(3)

        # Apply low-pass filter for smoothing
        self._filtered_force = (
            self._force_filter_alpha * scaled_force +
            (1 - self._force_filter_alpha) * self._filtered_force
        )

        # Saturate to max force
        filtered_magnitude = float(np.linalg.norm(self._filtered_force))
        if filtered_magnitude > self.max_force:
            self._filtered_force = (
                self._filtered_force / filtered_magnitude * self.max_force
            )

        self.state.force_output = self._filtered_force

    def render_vibration(self, frequency: float, amplitude: float, duration: float) -> None:
        """
        Generate vibration feedback.

        Args:
            frequency: Vibration frequency in Hz
            amplitude: Vibration amplitude (0-1)
            duration: Duration in seconds
        """
        # In a real implementation, this would command the haptic device
        # to generate vibrotactile feedback
        pass

    def get_workspace_position(self) -> NDArray[np.float64]:
        """Get normalized position within workspace (-1 to 1)."""
        return self.state.position / self.workspace_radius


class HapticFeedbackController:
    """
    Controller for haptic feedback rendering.

    Processes force sensor data and renders appropriate feedback
    to the haptic devices on the master console.
    """

    def __init__(self) -> None:
        self.devices: dict[str, HapticDevice] = {}
        self.force_sensors: dict[int, ForceReading] = {}

        # Feedback parameters
        self.global_scaling = 1.0
        self.tissue_stiffness_map: dict[str, float] = {
            "soft_tissue": 100.0,    # N/m
            "muscle": 500.0,
            "cartilage": 2000.0,
            "bone": 10000.0,
        }

        # Safety limits
        self.max_interaction_force = 10.0  # Newtons
        self._force_limit_callbacks: list[Callable[[int, float], None]] = []

    def add_device(self, device: HapticDevice) -> None:
        """Register a haptic device."""
        self.devices[device.name] = device

    def update_force_sensor(self, arm_index: int, reading: ForceReading) -> None:
        """Update force sensor reading from an arm."""
        self.force_sensors[arm_index] = reading

        # Check force limits
        if reading.force_magnitude > self.max_interaction_force:
            for callback in self._force_limit_callbacks:
                callback(arm_index, reading.force_magnitude)

    def register_force_limit_callback(
        self, callback: Callable[[int, float], None]
    ) -> None:
        """Register callback for force limit exceeded events."""
        self._force_limit_callbacks.append(callback)

    def compute_feedback(
        self,
        arm_index: int,
        device_name: str,
        tissue_type: str = "soft_tissue"
    ) -> NDArray[np.float64]:
        """
        Compute haptic feedback force for a device.

        Args:
            arm_index: Index of robot arm
            device_name: Name of haptic device
            tissue_type: Type of tissue being contacted

        Returns:
            3D force vector for haptic rendering
        """
        if arm_index not in self.force_sensors:
            return np.zeros(3)

        if device_name not in self.devices:
            return np.zeros(3)

        sensor_reading = self.force_sensors[arm_index]
        device = self.devices[device_name]

        # Scale forces based on tissue type
        stiffness_factor = self.tissue_stiffness_map.get(tissue_type, 100.0) / 100.0

        # Compute feedback force
        feedback_force = (
            sensor_reading.forces *
            self.global_scaling *
            stiffness_factor *
            device.force_scaling
        )

        return feedback_force

    def update(self) -> None:
        """
        Main update loop - compute and apply feedback to all devices.
        Should be called at haptic control rate (typically 1kHz).
        """
        for device_name, device in self.devices.items():
            if not device.is_enabled:
                continue

            # Find corresponding arm (simple mapping by index)
            arm_mapping = {"left": 0, "right": 1}
            arm_index = arm_mapping.get(device_name, -1)

            if arm_index >= 0 and arm_index in self.force_sensors:
                feedback = self.compute_feedback(arm_index, device_name)
                device.set_force(feedback)


class GraspingForceEstimator:
    """
    Estimates grasping forces for instruments without direct force sensing.

    Uses motor current and grip angle to estimate interaction forces.
    """

    def __init__(self) -> None:
        self.motor_constant = 0.1  # Nm/A
        self.gear_ratio = 100.0
        self.jaw_length = 0.01  # meters

        # Calibration parameters
        self.friction_compensation = 0.5  # Newtons
        self.stiffness_estimate = 1000.0  # N/m

    def estimate_from_current(
        self,
        motor_current: float,
        grip_angle: float
    ) -> float:
        """
        Estimate grasping force from motor current.

        Args:
            motor_current: Motor current in Amps
            grip_angle: Gripper angle in radians

        Returns:
            Estimated grip force in Newtons
        """
        # Motor torque
        motor_torque = motor_current * self.motor_constant

        # Output torque after gearbox
        output_torque = motor_torque * self.gear_ratio

        # Force at jaw tips
        tip_force = output_torque / self.jaw_length

        # Compensate for friction
        estimated_force = max(0, tip_force - self.friction_compensation)

        return estimated_force

    def estimate_from_deflection(
        self,
        nominal_angle: float,
        actual_angle: float
    ) -> float:
        """
        Estimate grasping force from jaw deflection.

        Args:
            nominal_angle: Commanded grip angle
            actual_angle: Measured grip angle

        Returns:
            Estimated contact force in Newtons
        """
        deflection = abs(nominal_angle - actual_angle)
        estimated_force = deflection * self.stiffness_estimate

        return estimated_force


@dataclass
class TissueInteractionModel:
    """Model for simulating tissue interaction forces."""
    stiffness: float = 500.0  # N/m
    damping: float = 10.0  # Ns/m
    friction_coefficient: float = 0.3
    max_deformation: float = 0.01  # meters

    def compute_contact_force(
        self,
        penetration_depth: float,
        velocity: NDArray[np.float64],
        normal: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute contact force based on penetration.

        Args:
            penetration_depth: Depth of penetration into tissue (m)
            velocity: Velocity of instrument tip (m/s)
            normal: Surface normal direction

        Returns:
            Contact force vector
        """
        if penetration_depth <= 0:
            return np.zeros(3)

        # Clamp penetration
        depth = min(penetration_depth, self.max_deformation)

        # Spring-damper model
        normal = normal / np.linalg.norm(normal)
        normal_velocity = np.dot(velocity, normal)

        spring_force = self.stiffness * depth
        damper_force = self.damping * abs(normal_velocity)

        total_normal_force = spring_force + damper_force

        # Add friction (tangential)
        tangent_velocity = velocity - normal_velocity * normal
        tangent_speed = float(np.linalg.norm(tangent_velocity))

        friction_force = np.zeros(3)
        if tangent_speed > 0.001:
            friction_direction = -tangent_velocity / tangent_speed
            friction_force = (
                self.friction_coefficient * total_normal_force * friction_direction
            )

        return total_normal_force * normal + friction_force
